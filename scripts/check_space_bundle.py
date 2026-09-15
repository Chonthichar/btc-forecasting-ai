"""Fail a Space build if the dashboard or registered model files were omitted."""
from pathlib import Path
import yaml


def check_bundle(root):
    required = [
        "app/space_server.py", "app/api.py", "app/analyst_api.py",
        "app/agents/orchestrator.py", "app/agents/decision_agent.py",
        "app/agents/forecast_context_agent.py", "app/agents/market_research_agent.py",
        "app/agents/validation_agent.py", "app/agents/answer_guard.py",
        "app/services/llm_service.py", "app/services/agent_config.py",
        "app/services/tavily_service.py", "app/schemas/agent_models.py",
        "app/services/reasoning_agents.py", "app/services/research_workflow.py", "app/services/market_decision.py",
        "app/services/research_triggers.py", "app/services/market_monitoring.py",
        "app/services/monitoring_events_store.py", "app/services/monitoring_backup.py",
        "app/services/monitor_schedule.py", "app/services/monitor_worker_runtime.py",
        "app/static/monitor/monitoring-events.js",
        "app/static/monitor/index.html", "app/static/monitor/analyst.js",
        "app/static/monitor/market-overview.js", "app/static/monitor/light-overrides.css",
        "app/static/monitor/analyst.css", "app/static/monitor/dashboard.js",
        "app/static/monitor/styles.css", "app/static/monitor/store.js",
        "app/static/monitor/sidebar.js", "app/static/monitor/views.css",
        "app/static/monitor/workspace.css", "app/asset_catalog.yaml",
        "app/static/monitor/deployment.js", "requirements-agents.txt",
    ]
    # Frozen deployment models. A missing artifact here means /predictions
    # returns 503 in production, so fail the build instead.
    for horizon in (1, 6, 24):
        required += [
            f"btc_deployment_models/risk/risk_{horizon}h_logistic.joblib",
            f"btc_deployment_models/risk/risk_{horizon}h_scaler.joblib",
            f"btc_deployment_models/direction/direction_{horizon}h_transformer.pt",
            f"btc_deployment_models/direction/direction_{horizon}h_scaler.joblib",
        ]
    required += [
        "btc_deployment_models/risk/risk_feature_list.csv",
        "btc_deployment_models/risk/risk_band_empirical_results.csv",
        "btc_deployment_models/risk/provisional_risk_band_thresholds.json",
        "btc_deployment_models/direction/direction_stationary_feature_list.csv",
    ]
    registry = yaml.safe_load((root / "app/deployment_registry.yaml").read_text(encoding="utf-8"))
    for spec in registry.values():
        directory = Path(spec["output_subdir"])
        required.extend([spec["config"], directory / "model_state.pt"])
        if spec["kind"] in {"lstm", "gru"}:
            required.append(directory / "scaler.joblib")
        elif spec["kind"] == "vae_transformer_v2":
            required.append(directory / "market_scaler.joblib")
            if spec["feature_set"] == "market_plus_sentiment":
                required.append(directory / "sentiment_scaler.joblib")
        else:
            raise SystemExit("Unsupported model kind in deployment registry")
    missing = [str(item) for item in required if not (root / item).is_file() or (root / item).stat().st_size == 0]
    if missing:
        raise SystemExit("Incomplete Space upload. Missing required files:\n" + "\n".join(missing))
    print(f"Space bundle complete: dashboard and {len(registry)} trained horizon models.")


if __name__ == "__main__":
    check_bundle(Path(__file__).resolve().parents[1])
