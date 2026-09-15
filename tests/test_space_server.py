import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

class SpaceTests(unittest.TestCase):
    def test_single_port_serves_dashboard_api_and_gradio(self):
        from app import api
        from app.monitoring_service import MonitoringService
        from app.space_server import create_app
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {"SPACE_START_MONITOR":"false", "PORT":"7860", "BTC_MONITOR_DB":str(Path(temporary)/"monitor.sqlite3")}), patch.object(api, "_service", None), patch.object(api, "_analyst", None):
                public = create_app()
                with TestClient(public) as client:
                    self.assertEqual(client.get("/",follow_redirects=False).headers["location"],"/monitor")
                    page = client.get("/monitor").text
                    self.assertIn("AI market analyst", page)
                    for view in ("forecast", "analyst", "performance", "method"):
                        self.assertIn(f"id=\"view-{view}\"", page)
                    self.assertIn("id=\"horizon-bars\"", page)
                    # Held-out performance is reported alongside the live record so a
                    # small live sample cannot be read as established skill.
                    self.assertIn("id=\"holdout-grid\"", page)
                    self.assertIn("id=\"sample-note\"", page)
                    self.assertNotIn("id=\"hero-card\"", page)
                    self.assertIn("id=\"agent-review\"", page)
                    self.assertIn("<h1>Market overview</h1>", page)
                    self.assertIn("id=\"asset-cards\"", page)
                    self.assertIn("data-asset-field=\"name\"", page)
                    # One horizon control drives every view; duplicates previously
                    # had to be resynchronised with a simulated click.
                    self.assertEqual(page.count("data-horizon="), 3)
                    # The asset selector is built from /assets, never hardcoded,
                    # and a coin is only deployed when its artifacts loaded.
                    self.assertNotIn("data-asset=\"ETH\"", page)
                    catalog = client.get("/assets").json()["assets"]
                    self.assertEqual([item["code"] for item in catalog], ["BTC", "ETH", "ADA"])
                    self.assertTrue(catalog[0]["deployed"])
                    self.assertEqual(catalog[0]["horizons"], [1, 6, 24])
                    self.assertFalse(any(item["deployed"] for item in catalog[1:]))
                    self.assertTrue(all(item["horizons"] == [] for item in catalog[1:]))
                    # Asset URLs are stamped with a hash of the asset contents, so a
                    # browser cannot serve a previous build from cache. Hand-written
                    # version strings were forgotten between deploys.
                    stamps = set(re.findall(r"/monitor/assets/[\w.-]+\.(?:js|css)\?v=([^\"']+)", page))
                    self.assertEqual(len(stamps), 1, f"assets must share one stamp, got {stamps}")
                    self.assertRegex(stamps.pop(), r"^[0-9a-f]{12}$")
                    self.assertNotIn("?v=2026", page)
                    self.assertEqual(client.get("/monitor/assets/analyst.js").status_code,200)
                    self.assertEqual(client.get("/monitor/assets/market-overview.js").status_code,200)
                    self.assertEqual(client.get("/monitor/assets/light-overrides.css").status_code,200)
                    self.assertEqual(client.get("/context").status_code,200)
                    self.assertEqual(client.get("/health").status_code,200)
                    self.assertEqual(client.get("/gradio/").status_code,200)

    def test_worker_uses_public_port_without_provider_credentials_and_stops(self):
        from app.space_server import create_app
        worker = Mock()
        worker.poll.return_value = None
        with patch.dict(os.environ, {"SPACE_START_MONITOR":"true", "PORT":"7860", "OPENAI_API_KEY":"fixture-key", "TAVILY_API_KEY":"fixture-key"}), patch("app.space_server.subprocess.Popen",return_value=worker) as launch:
            with TestClient(create_app()):
                environment = launch.call_args.kwargs["env"]
                self.assertEqual(environment["BTC_API_URL"],"http://127.0.0.1:7860")
                self.assertNotIn("OPENAI_API_KEY",environment)
                self.assertNotIn("TAVILY_API_KEY",environment)
            worker.terminate.assert_called_once()
            worker.wait.assert_called_once()

if __name__ == "__main__":
    unittest.main()
