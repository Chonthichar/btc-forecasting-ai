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
        "app/static/monitor/index.html", "app/static/monitor/analyst.js",
        "app/static/monitor/analyst.css", "app/static/monitor/dashboard.js",
        "app/static/monitor/styles.css", "requirements-agents.txt",
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
