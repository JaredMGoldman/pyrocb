import os
import sqlite3
import pickle

class SignalCache:
    def __init__(self, db_path="sounding_pipeline_cache.db", overwrite=False):
        self.db_path = db_path
        
        # Ensure parent directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        # Only wipe disk cache if explicitly requested
        if overwrite and os.path.exists(db_path):
            self._purge_db_files()

        self._init_db()

    def _init_db(self):
        """Initializes connection and ensures table schema exists."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=OFF;")
        conn.execute("CREATE TABLE IF NOT EXISTS soundings (id INTEGER PRIMARY KEY, payload BLOB);")
        conn.commit()
        conn.close()

    @classmethod
    def load(cls, db_path):
        """
        Loads an existing cache file. Raises FileNotFoundError if missing.
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"No cache file found at: '{db_path}'")
        return cls(db_path=db_path, overwrite=False)

    @classmethod
    def get_or_create(cls, db_path):
        """
        Factory method: Loads an existing cache if found, otherwise initializes a new one.
        """
        return cls(db_path=db_path, overwrite=False)

    def count(self):
        """Returns total record count in the cache."""
        if not os.path.exists(self.db_path):
            return 0
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM soundings;")
        total = cursor.fetchone()[0]
        conn.close()
        return total

    def append_batch(self, batch):
        if not batch:
            return
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=OFF;")
        cursor = conn.cursor()
        cursor.executemany("INSERT INTO soundings (payload) VALUES (?);", [(pickle.dumps(obj),) for obj in batch])
        conn.commit()
        conn.close()

    def stream_records(self, chunk_size=50000):
        if not os.path.exists(self.db_path):
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT payload FROM soundings;")
        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break
            for row in rows:
                yield pickle.loads(row[0])
        conn.close()

    def _purge_db_files(self):
        """Internal helper to clean up database and WAL artifacts."""
        if os.path.exists(self.db_path):
            try:
                # Reset WAL journal mode to allow clean file removal
                conn = sqlite3.connect(self.db_path)
                conn.execute("PRAGMA journal_mode=DELETE;")
                conn.close()
            except Exception:
                pass

        for ext in ["", "-wal", "-shm"]:
            path = self.db_path + ext
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def destroy(self):
        """Cleans up the scratchpad database file completely from disk."""
        self._purge_db_files()