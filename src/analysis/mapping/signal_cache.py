import os
import sqlite3
import pickle

class SignalCache:
    def __init__(self, db_path="sounding_pipeline_cache.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok = True) 
        # Clean up old caches if they exist from an interrupted run
        if os.path.exists(db_path):
            try: os.remove(db_path)
            except: pass
    
        conn = sqlite3.connect(self.db_path)
        # WAL mode + synchronous OFF allows lightning fast sequential append speeds
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=OFF;")
        conn.execute("CREATE TABLE IF NOT EXISTS soundings (id INTEGER PRIMARY KEY, payload BLOB);")
        conn.commit()
        conn.close()

    def append_batch(self, batch):
        if not batch: return
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=OFF;")
        cursor = conn.cursor()
        # Binary pickle blob storage avoids heavy object overhead constraints
        cursor.executemany("INSERT INTO soundings (payload) VALUES (?);", [(pickle.dumps(obj),) for obj in batch])
        conn.commit()
        conn.close()

    def stream_records(self, chunk_size=50000):
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
        
    def destroy(self):
        # Cleans up the scratchpad database file cleanly from disk when pipeline completes
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=DELETE;") # resets WAL file
        conn.close()
        if os.path.exists(self.db_path): os.remove(self.db_path)
        if os.path.exists(self.db_path + "-wal"): os.remove(self.db_path + "-wal")
        if os.path.exists(self.db_path + "-shm"): os.remove(self.db_path + "-shm")