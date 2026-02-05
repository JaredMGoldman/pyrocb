from __future__ import annotations

import fnmatch
import os
from pathlib import Path

import paramiko


HOST = "fuoco.geog.umd.edu"
USER = "fire"
PASSWORD = "burnt"

def modis_client(pattern: str, local_dir: str) -> list[str]:
    """
    Download all files in remote_dir whose basename matches pattern (e.g. '*.hdf' or 'MCD14ML.*.hdf').
    Returns list of local file paths.
    """
    local_dir_p = Path(local_dir)
    local_dir_p.mkdir(parents=True, exist_ok=True)

    transport = paramiko.Transport((HOST, 22))
    transport.connect(username=USER, password=PASSWORD)

    sftp = paramiko.SFTPClient.from_transport(transport)
    try:
        sftp.chdir("/data/MODIS/C6/MCD14ML")
        names = sftp.listdir(".")
        matches = [n for n in names if fnmatch.fnmatch(n, pattern)]

        out_paths = []
        for name in matches:
            local_path = local_dir_p / name
            sftp.get(name, str(local_path))
            out_paths.append(str(local_path))
        return out_paths
    finally:
        sftp.close()
        transport.close()


# Example usage (edit remote path to what you need)
downloaded = modis_client(
    pattern="*",
    out_dir="MODIS_MCD14ML",
)
print(f"Downloaded {len(downloaded)} files")
