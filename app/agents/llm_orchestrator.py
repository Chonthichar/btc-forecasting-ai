from __future__ import annotations
import json
import os

class CryptoLLMOrchestrator:
    """OpenAI Responses API orchestration; forecasting numbers come from ML tools."""

    def __init__(self, tool_impl, model=None, api_key=None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1")
        self.tool_impl = tool_impl

        self.tools = [
            {
                "type": "function",
                "name": "get_latest_market",
                "description": "Return latest closed BTC market snapshot from Binance.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                "strict": True,
            },
            {
                "type": "function",
                "name": "get_latest_sentiment",
                "description": "Return latest BTC news sentiment and coverage.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                "strict": True,
            },
            {
                "type": "function",
                "name": "get_forecast",
                "description": "Return trained ML model BTC direction probability.",
                "parameters": {
                    "type": "object",
                    "properties": {"horizon_hours": {"type": "integer", "enum": [1, 6, 24]}},
                    "required": ["horizon_hours"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "get_system_health",
                "description": "Check data continuity, freshness and feature completeness.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                "strict": True,
            },
            {
                "type": "function",
                "name": "refresh_live_data",
                "description": "Refresh live market, sentiment, features and forecasts.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                "strict": True,
            },
        ]
        self.instructions = """
You are the orchestration/explanation layer for an academic BTC forecasting prototype.
Never invent market values, sentiment values, probabilities, metrics, or timestamps.
For forecasting questions call get_system_health and get_forecast.
For current/latest/now questions call refresh_live_data first.
The trained ML model, not the LLM, produces the numerical forecast.
Explain probabilities as uncertain signals, not guarantees.
Do not give personalized buy/sell instructions.
"""

    def ask(self, question):
        response = self.client.responses.create(
            model=self.model,
            instructions=self.instructions,
            input=question,
            tools=self.tools,
        )
        for _ in range(8):
            calls = [x for x in response.output if getattr(x, "type", None) == "function_call"]
            if not calls:
                return response.output_text

            outputs = []
            for call in calls:
                args = json.loads(call.arguments or "{}")
                result = self.tool_impl[call.name](**args)
                outputs.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result, default=str),
                })

            response = self.client.responses.create(
                model=self.model,
                previous_response_id=response.id,
                instructions=self.instructions,
                input=outputs,
                tools=self.tools,
            )
        raise RuntimeError("Maximum tool-call iterations exceeded.")
