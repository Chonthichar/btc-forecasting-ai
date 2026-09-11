import os
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
                    self.assertIn("AI Market Analyst", page)
                    self.assertIn("data-asset=\"BTC\"", page)
                    self.assertIn("data-asset=\"ETH\"", page)
                    self.assertIn("data-asset=\"ADA\"", page)
                    self.assertIn("id=\"agent-review\"", page)
                    self.assertIn("<h1>Market overview</h1>", page)
                    self.assertIn("<h2>News analysis</h2>", page)
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
