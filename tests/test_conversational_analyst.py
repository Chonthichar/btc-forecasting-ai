import unittest
import test_analyst_integration as integration
from test_analyst_integration import PlanStub, ResearchStub, article, saved_snapshot
from app.schemas.agent_models import DecisionPlan, ResearchResult

def plan(answer, facts=None, sources=None, interpretation="model_only"):
    return DecisionPlan(focus_horizons=["1h"], fact_ids=facts or [], evidence_ids=sources or [],
        interpretation=interpretation, answer_style="concise", answer=answer)

class ConversationalTests(unittest.TestCase):
    def build(self, **kwargs):
        return integration.AnalystIntegrationTests().build(**kwargs)

    def test_greeting_is_model_written_without_market_dump_or_reliability_notice(self):
        selected = plan("Hi! What would you like to explore about Bitcoin?", interpretation="insufficient_evidence")
        client, _, _, research = self.build(llm=PlanStub(selected))
        response = client.post("/chat", json={"message":"hello"}).json()
        self.assertEqual(response["answer"], selected.answer)
        self.assertEqual(response["llm_status"], "ok")
        self.assertNotIn("Model view", response["answer"])
        self.assertNotIn("65,000", response["answer"])
        self.assertNotIn("No reliable forecast", response["answer"])
        self.assertEqual(research.calls, [])

    def test_question_specific_prose_and_exact_model_fact_are_both_rendered(self):
        selected = plan("Here is the model behind that horizon: vae_transformer_v2. {{fact:1h.model}}\n\nIts raw output and its reliability are different things.", ["1h.model"])
        client, _, _, _ = self.build(llm=PlanStub(selected))
        result = client.post("/chat", json={"message":"What model is used for 1h?"}).json()
        self.assertEqual(result["llm_status"], "ok")
        self.assertIn("Here is the model behind that horizon", result["answer"])
        self.assertIn("vae_transformer_v2", result["answer"])
        self.assertIn("No reliable forecast", result["answer"])
        self.assertNotIn("Current market context:", result["answer"])
        self.assertNotIn("{{", result["answer"])

    def test_follow_up_history_reaches_the_single_decision_call(self):
        llm = PlanStub(plan("It measures a different property: direction describes the predicted move, while reliability describes how much the validated signal can be trusted.", interpretation="insufficient_evidence"))
        client, _, _, _ = self.build(llm=llm)
        result = client.post("/chat", json={"message":"Can you explain that difference?", "history":[
            {"role":"user","content":"What is reliability?"},
            {"role":"assistant","content":"Reliability differs from directional probability."}]}).json()
        self.assertEqual(result["llm_status"], "ok")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0][1][0].content, "What is reliability?")
        self.assertNotIn("closing price", result["answer"])

    def test_invented_values_urls_and_references_fall_back(self):
        for text in ("BTC is $999999. {{fact:1h.raw}}", "{{fact:made.up}}", "Read https://fake.invalid {{fact:1h.raw}}", "{{fact:1h.raw}} {{bad:1h.raw}}"):
            with self.subTest(text=text):
                client, _, _, _ = self.build(llm=PlanStub(plan(text,["1h.raw"])))
                result = client.post("/chat",json={"message":"Explain 1h forecast"}).json()
                self.assertEqual(result["llm_status"], "invalid_output")
                self.assertNotIn("AI answer could not be verified", result["answer"])
                self.assertTrue(result["answer"].strip())
                self.assertNotIn("999999", result["answer"])
                self.assertNotIn("fake.invalid", result["answer"])
                self.assertNotIn("{{", result["answer"])

    def test_unknown_reliability_cannot_be_promoted_by_prose(self):
        for text in ("This is a reliable bearish signal. {{fact:1h.raw}}", "Bitcoin will definitely rise. {{fact:1h.raw}}", "The model predicts lower prices because ETF outflows increased. {{fact:1h.raw}}"):
            with self.subTest(text=text):
                client, _, _, _ = self.build(llm=PlanStub(plan(text,["1h.raw"])))
                result = client.post("/chat",json={"message":"Explain 1h forecast"}).json()
                self.assertEqual(result["llm_status"], "invalid_output")
                self.assertEqual(result["signal_state"], "UNKNOWN")
                self.assertNotIn(text.split(" {{")[0], result["answer"])

    def test_reliable_down_is_preserved_in_conversational_answer(self):
        selected = plan("The raw output favors the downward direction, with P(UP) = 10.0%. {{fact:1h.raw}}",["1h.raw"])
        client, _, _, _ = self.build(snapshot=saved_snapshot(reliability=.8,p=.1),llm=PlanStub(selected))
        result = client.post("/chat",json={"message":"Explain 1h forecast"}).json()
        self.assertEqual(result["signal_state"], "DOWN")
        self.assertEqual(result["llm_status"], "ok")
        self.assertIn("P(UP) = 10.0%", result["answer"])
        self.assertIn("80.0%", result["answer"])

    def test_mixed_news_requires_opposing_sources_in_actual_answer(self):
        result = ResearchResult(status="ok",evidence=[article(),article("bear",headline="Bitcoin ETF outflows rise",direction="bearish")])
        for answer, expected in (("The evidence points in different directions. {{source:bull}} {{source:bear}}", "ok"),
                                 ("The evidence is bullish. {{source:bull}}", "invalid_output")):
            client, _, _, _ = self.build(research=ResearchStub(result),llm=PlanStub(plan(answer,sources=["bull","bear"],interpretation="mixed_context")))
            response = client.post("/chat",json={"message":"What are the bullish and bearish news factors?"}).json()
            self.assertEqual(response["llm_status"],expected)
            self.assertIn("BULLISH",response["answer"])
            self.assertIn("BEARISH",response["answer"])

    def test_missing_openai_greeting_is_safe_and_not_a_forecast_report(self):
        client, _, _, _ = self.build()
        result = client.post("/chat",json={"message":"hi"}).json()
        self.assertEqual(result["llm_status"],"unavailable")
        self.assertIn("OpenAI is not configured",result["answer"])
        self.assertNotIn("closing price",result["answer"])

    def test_repeated_cited_probability_and_horizon_labels_are_not_false_rejections(self):
        selected = plan("The 1-hour model assigns 10.0% to an upward move. {{fact:1h.raw}}",["1h.raw"])
        client, _, _, _ = self.build(llm=PlanStub(selected))
        result = client.post("/chat",json={"message":"Explain 1h forecast"}).json()
        self.assertEqual(result["llm_status"],"ok")
        self.assertIn("1-hour",result["answer"])
        self.assertIn("10.0%",result["answer"])

    def test_conceptual_followup_can_use_model_only_without_dumping_selected_facts(self):
        selected = plan("It describes the predicted direction, rather than independently validated reliability.")
        client, _, _, _ = self.build(llm=PlanStub(selected))
        result = client.post("/chat",json={"message":"Explain that more simply"}).json()
        self.assertEqual(result["llm_status"],"ok")
        self.assertEqual(result["answer"],selected.answer)

    def test_fact_citations_are_hidden_without_repeating_catalog_sentences(self):
        selected = plan("It predicts direction; that is different from validated reliability. {{fact:concept.probability_up}}", ["concept.probability_up"])
        client, _, _, _ = self.build(llm=PlanStub(selected))
        result = client.post("/chat", json={"message":"What does probability_up mean?"}).json()
        self.assertEqual(result["llm_status"],"ok")
        self.assertEqual(result["answer"],"It predicts direction; that is different from validated reliability.")

    def test_shorthand_citations_accept_only_known_selected_facts(self):
        for citation, expected in (("{{concept.reliability}}", "ok"), ("{{invented.reliability}}", "invalid_output")):
            selected = plan("Direction and validated reliability measure different things. " + citation,["concept.reliability"])
            client, _, _, _ = self.build(llm=PlanStub(selected))
            result = client.post("/chat",json={"message":"What does reliability mean?"}).json()
            self.assertEqual(result["llm_status"],expected)
            self.assertNotIn("{{",result["answer"])

    def test_general_reliability_definition_is_not_a_signal_promotion(self):
        selected = plan("A reliable forecast requires separate validation. {{concept.reliability}}",["concept.reliability"])
        client, _, _, _ = self.build(llm=PlanStub(selected))
        result = client.post("/chat",json={"message":"What does reliability mean?"}).json()
        self.assertEqual(result["llm_status"],"ok")
        self.assertEqual(result["signal_state"],"UNKNOWN")

    def test_reliability_fallback_answers_the_question_without_a_price_dump(self):
        client, _, _, _ = self.build()
        result = client.post("/chat",json={"message":"Why is reliability unknown?"}).json()
        self.assertEqual(result["llm_status"],"unavailable")
        self.assertIn("validated reliability",result["answer"])
        self.assertNotIn("closing price",result["answer"])
        self.assertNotIn("P(UP) =",result["answer"])

    def test_news_numbers_must_be_present_in_the_cited_source(self):
        research = ResearchResult(status="ok",evidence=[article(summary="The reported Bitcoin fund inflow was 20% higher.")])
        for value, expected in (("20%","ok"),("90%","invalid_output")):
            selected = plan("The cited report describes inflows as " + value + " higher. {{source:bull}}",sources=["bull"],interpretation="bullish_context")
            client, _, _, _ = self.build(research=ResearchStub(research),llm=PlanStub(selected))
            result = client.post("/chat",json={"message":"Show latest Bitcoin news"}).json()
            self.assertEqual(result["llm_status"],expected)

    def test_news_fallback_keeps_news_focus_without_forecast_boilerplate(self):
        research = ResearchResult(status="ok",evidence=[article()])
        client, _, _, _ = self.build(research=ResearchStub(research))
        result = client.post("/chat",json={"message":"Show latest Bitcoin market news"}).json()
        self.assertIn("Bitcoin ETF inflows",result["answer"])
        self.assertNotIn("closing price",result["answer"])
        self.assertNotIn("P(UP) =",result["answer"])

if __name__ == "__main__":
    unittest.main()
