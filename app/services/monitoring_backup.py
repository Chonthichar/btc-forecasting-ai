"""Optional private Hub snapshots for Spaces with ephemeral local disks.

SQLite remains on local disk. Only consistent backup files are uploaded;
we never run SQLite/WAL directly on an object-storage mount.
"""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import threading
from contextlib import closing
from datetime import datetime, timezone


class MonitoringBackup:
    def __init__(self, repo=None, client=None, downloader=None):
        self.repo = repo if repo is not None else os.getenv("MONITOR_BACKUP_REPO", "").strip()
        self.token = os.getenv("MONITOR_BACKUP_TOKEN") or os.getenv("HF_TOKEN")
        self.client = client
        self.downloader = downloader
        self.revision = None
        self.lock = threading.Lock()

    def api(self):
        if self.client is None:
            from huggingface_hub import HfApi
            self.client = HfApi(token=self.token)
        return self.client

    def restore(self, path):
        path = Path(path)
        if not self.repo or (path.exists() and path.stat().st_size):
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            info = self.api().repo_info(self.repo, repo_type="dataset")
            if not info.private:
                raise RuntimeError("Monitoring backup repository must be private")
            self.revision = info.sha
            download = self.downloader
            if download is None:
                from huggingface_hub import hf_hub_download
                download = hf_hub_download
            source = download(repo_id=self.repo, filename="monitoring.sqlite3", repo_type="dataset",
                              revision=self.revision, token=self.token, etag_timeout=15)
            # Validate before replacing anything; a failed restore cannot create
            # an empty database that would later overwrite the durable history.
            with closing(sqlite3.connect(f"file:{Path(source).as_posix()}?mode=ro", uri=True)) as db:
                if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise RuntimeError("Invalid monitoring backup")
                db.execute("SELECT id,probability_up,issued_at FROM forecasts LIMIT 1")
            temporary = path.with_suffix(".restoring")
            shutil.copyfile(source, temporary)
            os.replace(temporary, path)
            return True
        except Exception:
            raise RuntimeError("Monitoring history restore failed. Check private backup access; refusing to start with an empty history.") from None

    def save(self, store):
        if not self.repo or not self.lock.acquire(blocking=False):
            return
        try:
            if self.revision is None:
                info = self.api().repo_info(self.repo, repo_type="dataset")
                if not info.private:
                    raise RuntimeError("Monitoring backup must be private")
                self.revision = info.sha
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "monitoring.sqlite3"
                with store._connection() as source, closing(sqlite3.connect(target)) as destination:
                    source.backup(destination)
                    # A downloadable snapshot must be self-contained; source
                    # WAL mode otherwise leaves cleanup writes in a sidecar.
                    destination.execute("PRAGMA journal_mode=DELETE")
                    # Runtime ownership cannot survive a container replacement.
                    destination.execute("DELETE FROM worker_lock")
                    destination.execute("DELETE FROM operation_leases")
                    destination.execute("DELETE FROM state WHERE key IN ('worker','active_run')")
                    destination.execute("UPDATE research_events SET status='error' WHERE status='running'")
                    destination.commit()
                commit = self.api().upload_file(path_or_fileobj=str(target), path_in_repo="monitoring.sqlite3",
                    repo_id=self.repo, repo_type="dataset", parent_commit=self.revision,
                    commit_message="Save monitoring history and event state")
                self.revision = commit.oid
            store.set_state("backup", {"status": "OK", "saved_at": datetime.now(timezone.utc).isoformat()})
        except Exception:
            store.set_state("backup", {"status": "ERROR", "error": "Durable backup unavailable; local history remains saved. Check access or conflicting writers."})
        finally:
            self.lock.release()
