# Single-port Docker startup

The default image command now serves the dashboard and API on port **7860**, with Gradio at `/gradio`. `/` redirects to `/monitor`. Chat uses the same origin, so it does not try to reach a visitor's `localhost:8000`.

Keep the existing README metadata:

```yaml
sdk: docker
app_port: 7860
```

On Hugging Face, set `OPENAI_API_KEY` and `TAVILY_API_KEY` in **Space Settings → Secrets**, and `OPENAI_MODEL=gpt-4.1` as a variable. Never upload `.env`. Docker Spaces support runtime secrets as environment variables and expose the configured app port. [Official Docker Spaces documentation](https://huggingface.co/docs/hub/en/spaces-sdks-docker)

The single-container entry point starts the hourly worker too; `SPACE_START_MONITOR=false` disables it for smoke tests. Configure `BTC_MONITOR_DB` on persistent storage if forecast history must survive Space restarts; default Space disk is not persistent. [Storage behavior](https://huggingface.co/docs/hub/en/spaces-sdks-docker#data-persistence)

Local Compose keeps the API on 8000, Gradio on 7860, and a separate worker. Explicit API command and healthcheck prevent the Space default from breaking this arrangement:

```powershell
cd 'C:\Users\Windows PC\Downloads\BTC_Forecasting_Production_First_Bundle\btc_prod'
docker compose --profile ui up --build -d
```

The complete release was built and tested locally as UID 1000. All three registered models and their scalers loaded offline. Remote verification is performed after the Space rebuilds.

## Avoid partial uploads

Upload the complete application folder, including `app/space_server.py`, `app/schemas`, `app/services`, analyst UI assets, and all model checkpoints/scalers referenced by `app/deployment_registry.yaml`. The Docker build now runs `scripts/check_space_bundle.py` to reject incomplete uploads. The `.gitignore` explicitly includes the seven required `.pt` and `.joblib` files.

The final Docker command must be `CMD ["python", "-m", "app.space_server"]`. Starting `app/app_gradio.py` directly serves the compact Gradio interface instead of the dashboard. Hugging Face runs this Docker command; local Compose has its own commands.

The deployment repair was uploaded to `chonthichar/crypto-forecasting-ai` at commit `c2769aff933c0c3a61b8d5d646f56913c8d587f7`. The existing Space selected `OPENAI_MODEL=gpt-5.6`; that setting and its provider secrets were preserved. The local setup can keep its own model selection.
