"""Stable entry point for the browser-independent monitoring worker."""
from .services.monitor_worker_runtime import MonitoringWorker, main, next_hourly_run

if __name__ == "__main__":
    main()
