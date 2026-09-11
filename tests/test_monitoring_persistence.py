import shutil
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from app.monitoring_store import MonitoringStore
from app.monitoring_service import MonitoringService
from app.services.monitoring_backup import MonitoringBackup
from app.services.monitoring_events_store import EventStore

ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.store = MonitoringStore(self.path / 'live.sqlite3')
        self.events = EventStore(self.store)
        self.store.set_state('preserve', {'probability_up': .4123})
        self.client = Mock()
        self.client.repo_info.return_value = SimpleNamespace(private=True, sha='revision-a')
        self.client.upload_file.return_value = SimpleNamespace(oid='revision-b')
        self.backup = MonitoringBackup('owner/private', self.client)

    def tearDown(self):
        self.temp.cleanup()

    def test_consistent_snapshot_restore_preserves_data_and_clears_ownership(self):
        self.store.try_acquire_lock('old')
        self.events.acquire('scheduler', 'old')
        self.store.set_state('worker', {'worker_id': 'old'})
        self.events.claim('fingerprint', 'candle', ['price'], 30)
        archive = self.path / 'archive.sqlite3'
        def upload(**kwargs):
            self.assertEqual(kwargs['parent_commit'], 'revision-a')
            shutil.copyfile(kwargs['path_or_fileobj'], archive)
            return SimpleNamespace(oid='revision-b')
        self.client.upload_file.side_effect = upload
        self.backup.save(self.store)
        self.assertEqual(self.store.get_state('backup')['status'], 'OK')
        restored = self.path / 'restored.sqlite3'
        restore = MonitoringBackup('owner/private', self.client, lambda **_: str(archive))
        self.assertTrue(restore.restore(restored))
        db = MonitoringStore(restored)
        self.assertEqual(db.get_state('preserve'), {'probability_up': .4123})
        self.assertIsNone(db.get_state('worker'))
        self.assertTrue(EventStore(db).acquire('scheduler', 'new'))
        self.assertEqual(EventStore(db).latest_event()['status'], 'error')
        self.assertTrue(db.try_acquire_lock('new'))
        with sqlite3.connect(restored) as connection:
            self.assertEqual(connection.execute('PRAGMA integrity_check').fetchone()[0], 'ok')

    def test_failed_upload_preserves_local_history_and_reports_error(self):
        self.client.upload_file.side_effect = RuntimeError('credential should not be exposed')
        self.backup.save(self.store)
        self.assertEqual(self.store.get_state('preserve')['probability_up'], .4123)
        status = self.store.get_state('backup')
        self.assertEqual(status['status'], 'ERROR')
        self.assertNotIn('credential', str(status))

    def test_public_repository_is_never_uploaded_to(self):
        self.client.repo_info.return_value.private = False
        self.backup.save(self.store)
        self.client.upload_file.assert_not_called()
        self.assertEqual(self.store.get_state('backup')['status'], 'ERROR')

    def test_corrupt_restore_fails_closed(self):
        corrupt = self.path / 'corrupt'
        corrupt.write_text('not a database')
        target = self.path / 'new.sqlite3'
        backup = MonitoringBackup('owner/private', self.client, lambda **_: str(corrupt))
        with self.assertRaisesRegex(RuntimeError, 'refusing to start'):
            backup.restore(target)
        self.assertFalse(target.exists())

    def test_existing_local_history_is_not_replaced(self):
        self.assertFalse(self.backup.restore(self.store.path))
        self.client.repo_info.assert_not_called()

    def test_disabled_backup_uses_no_network(self):
        backup = MonitoringBackup('', self.client)
        self.assertFalse(backup.restore(self.path / 'absent'))
        backup.save(self.store)
        self.client.repo_info.assert_not_called()

    def test_concurrent_workers_only_one_owns_scheduler(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda n: self.events.acquire('scheduler', str(n)), range(8)))
        self.assertEqual(sum(results), 1)

    def test_query_cache_expires_without_extending_on_reads(self):
        now = datetime.now(UTC)
        self.events.cache_put('query', {'results': []}, 600, now)
        self.assertIsNotNone(self.events.cache_get('query', now + timedelta(minutes=9)))
        self.assertIsNone(self.events.cache_get('query', now + timedelta(minutes=10)))

    def test_fingerprint_cooldown_survives_new_store_and_expires(self):
        now = datetime.now(UTC)
        _, first = self.events.claim('same', 'candle-a', [], 30, now)
        restarted = EventStore(MonitoringStore(self.store.path))
        _, recent = restarted.claim('same', 'candle-b', [], 30, now + timedelta(minutes=20))
        _, expired = restarted.claim('same', 'candle-b', [], 30, now + timedelta(minutes=31))
        self.assertEqual((first, recent, expired), (True, False, True))

    def test_finished_failure_keeps_retry_due_and_does_not_claim_success(self):
        service = MonitoringService(ROOT, self.path / 'service.sqlite3')
        now = datetime.now(UTC)
        started = now - timedelta(minutes=4)
        due = (now + timedelta(minutes=3)).isoformat()
        service.store.set_state('last_cycle_started', started.isoformat())
        service.store.set_state('last_cycle_finished', (started + timedelta(seconds=1)).isoformat())
        service.store.set_state('next_cycle_due', due)
        status = service.heartbeat(worker_id='new')
        self.assertEqual(status['next_cycle_due'], due)
        self.assertIsNone(status['last_cycle_completed'])

    def test_interrupted_cycle_is_rescheduled_after_lease_expires(self):
        service = MonitoringService(ROOT, self.path / 'service.sqlite3')
        now = datetime.now(UTC)
        service.store.set_state('last_cycle_started', (now - timedelta(minutes=4)).isoformat())
        service.store.set_state('next_cycle_due', (now + timedelta(minutes=30)).isoformat())
        status = service.heartbeat(worker_id='new')
        self.assertLess(datetime.fromisoformat(status['next_cycle_due']), now + timedelta(seconds=10))
        self.assertIsNone(status['last_cycle_completed'])
