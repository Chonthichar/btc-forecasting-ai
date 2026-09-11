"""Real SQLite/API tests; every OpenAI and Tavily boundary is offline/mocked."""
import asyncio
import copy
import json
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import test_event_monitoring as fixtures
from test_analyst_integration import settings, article, PlanStub, ResearchStub
from app import api
from app.agents.orchestrator import AgentOrchestrator, needs_research
from app.schemas.agent_models import (ResearchPlan, ReviewPlan, EvidenceReview, ResearchResult,
    DecisionPlan, DecisionProposal, ReviewReport)
from app.services.reasoning_agents import ReasoningAgents
from app.services.research_workflow import ResearchWorkflow
from app.services.market_decision import enforce_decision
from app.services.llm_service import LLMService


class ReasoningStub:
    def __init__(self):
        self.calls = []
        self.reject = set()
        self.use_cache = False
        self.repeats = 1
        self.extra_id = False
        self.fail = None
        self.more = False
        self.queries = ['Bitcoin market news ETF flows']

    def run(self, role, payload, search=None):
        self.calls.append((role, copy.deepcopy(payload)))
        if self.fail == role:
            return None, 'unavailable', True
        if role == 'Research':
            result = payload['cached_evidence']
            if not self.use_cache:
                first = search(self.queries)
                if 'evidence' in first:
                    result = first
                for _ in range(self.repeats - 1):
                    search(self.queries)
            return ResearchPlan(evidence_ids=[e['id'] for e in result.get('evidence', [])],
                                research_summary='Review the supplied reports independently.'), 'ok', True
        reviews = [EvidenceReview(evidence_id=e['id'], accept=e['id'] not in self.reject,
            stance=e['direction'].upper(), reason='Supported by supplied source text.' if e['id'] not in self.reject else 'Claim not supported.') for e in payload['evidence']]
        if self.extra_id:
            reviews.append(EvidenceReview(evidence_id='invented', accept=True, stance='BULLISH', reason='Untrusted claim'))
        return ReviewPlan(assessments=reviews, contradictions=[], unsupported_claims=list(self.reject),
            review_summary='Independent source assessment.', additional_research_needed=self.more), 'ok', True


class ThreeAgentTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.EventMonitoringTests()
        self.fixture.setUp()
        self.service = self.fixture.service
        self.settings = settings(openai_key='sk-unit-only', openai_model='gpt-5.6', tavily_key='tvly-unit-only')
        self.collector = ResearchStub(ResearchResult(status='ok', evidence=[article()],
            queries=['Bitcoin market news ETF flows'], retrieved_at=datetime.now(timezone.utc).isoformat(), provider_call_count=1))
        self.reasoning = ReasoningStub()
        self.workflow = ResearchWorkflow(self.service, self.settings, self.collector, self.reasoning)
        self.fixture.monitor.research = self.workflow
        self.service.store.save_snapshot(self.fixture.raw)
        self.decision = PlanStub(DecisionPlan(focus_horizons=['1h'], fact_ids=[], evidence_ids=['bull'],
            interpretation='bullish_context', answer_style='concise', answer='The report describes Bitcoin ETF flows. {{source:bull}}'))
        self.analyst = AgentOrchestrator(self.service, self.settings, research_agent=self.workflow, llm=self.decision)
        self.old_analyst = api._analyst
        api._analyst = self.analyst
        self.client = self.fixture.client

    def tearDown(self):
        api._analyst = self.old_analyst
        self.fixture.tearDown()

    def test_polling_and_pwa_routes_are_read_only(self):
        for _ in range(2):
            for route in ('/context', '/evidence', '/forecasts', '/monitor', '/monitoring/summary',
                          '/monitoring/history', '/monitoring/activity', '/monitoring/scans',
                          '/monitor/assets/manifest.json', '/monitor/assets/logo192.png'):
                response = self.client.get(route)
                self.assertEqual(response.status_code, 200, route)
        self.assertEqual(self.reasoning.calls, [])
        self.assertEqual(self.collector.calls, [])
        self.assertEqual(self.decision.calls, [])

    def test_normal_cycle_does_not_run_any_reasoning_agent(self):
        response = self.service.run()
        self.assertFalse(response['research_trigger']['should_research'])
        self.assertEqual(self.reasoning.calls, [])
        self.assertEqual(self.collector.calls, [])
        self.assertEqual(self.fixture.llm.calls, [])

    def test_trigger_runs_research_then_independent_review_and_decision(self):
        self.fixture.price_move()
        response = self.service.run()
        self.assertTrue(response['research_trigger']['should_research'])
        self.assertEqual([r for r, _ in self.reasoning.calls], ['Research', 'Review'])
        self.assertEqual(len(self.fixture.llm.calls), 1)
        event = self.service.event_store.latest_event()
        self.assertEqual(event['result']['workflow']['accepted_sources'], 1)
        self.assertEqual(event['result']['market_decision']['market_stance'], 'NO_RELIABLE_VIEW')

    def test_explicit_research_action_runs_workflow_and_additive_decision(self):
        response = self.client.post('/research', json={'fresh_research': True}).json()
        self.assertEqual([r for r, _ in self.reasoning.calls], ['Research', 'Review'])
        self.assertEqual(len(self.decision.calls), 1)
        self.assertTrue(response['research']['workflow']['user_requested'])
        self.assertTrue(response['research']['workflow']['tavily_called'])
        self.assertTrue(response['research']['workflow']['decision_agent_executed'])
        self.assertTrue(self.reasoning.calls[0][1]['search_required'])
        self.assertIn('validation', response)
        self.assertEqual(response['market_decision']['market_stance'], 'NO_RELIABLE_VIEW')

    def test_research_agent_can_choose_recent_cached_evidence(self):
        first = self.workflow.run('Research BTC', user_requested=True)
        self.service.store.set_state('analyst_evidence', first.model_dump())
        self.reasoning.use_cache = True
        second = self.workflow.run('Research BTC', user_requested=True)
        self.assertEqual(len(self.collector.calls), 1)
        self.assertTrue(second.workflow.cache_used)
        self.assertFalse(second.workflow.tavily_called)
        self.assertTrue(second.workflow.review_agent_executed)
        self.assertFalse(self.reasoning.calls[-2][1]['search_required'])

    def test_force_fresh_cannot_silently_reuse_old_evidence(self):
        first = self.workflow.run('Research BTC')
        self.service.store.set_state('analyst_evidence', first.model_dump())
        self.reasoning.use_cache = True
        second = self.workflow.run('Fresh BTC research', fresh=True)
        self.assertEqual(second.status, 'unavailable')
        self.assertEqual(second.evidence, [])

    def test_repeated_event_does_not_repeat_any_agent_or_tavily(self):
        self.fixture.price_move()
        self.service.run()
        count = len(self.reasoning.calls)
        self.service.run()
        self.assertEqual(len(self.reasoning.calls), count)
        self.assertEqual(len(self.collector.calls), 1)

    def test_review_rejects_unsupported_evidence_before_decision(self):
        self.reasoning.reject = {'bull'}
        response = self.client.post('/research', json={}).json()
        self.assertEqual(response['research']['evidence'], [])
        self.assertEqual(response['research']['review']['rejected_evidence'], ['bull'])
        self.assertEqual(response['research']['workflow']['rejected_sources'], 1)
        self.assertEqual(self.decision.calls, [])

    def test_python_rejects_future_source_before_semantic_review(self):
        self.collector.result.evidence.append(article('future', published_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat()))
        result = self.workflow.run('Research BTC')
        review_input = self.reasoning.calls[-1][1]['evidence']
        self.assertEqual([e['id'] for e in review_input], ['bull'])
        self.assertEqual(result.workflow.rejected_sources, 1)

    def test_reviewer_cannot_invent_or_restore_rejected_ids(self):
        self.reasoning.extra_id = True
        result = self.workflow.run('Research BTC')
        self.assertEqual(result.status, 'unavailable')
        self.assertEqual(result.evidence, [])

    def test_historical_cutoff_remains_authoritative(self):
        cutoff = datetime.now(timezone.utc) - timedelta(hours=4)
        self.collector.result.evidence[0].published_at = (cutoff + timedelta(hours=1)).isoformat()
        result = self.workflow.run('Historical BTC research', mode='historical', prediction_timestamp=cutoff)
        self.assertEqual(result.evidence, [])
        self.assertFalse(result.workflow.review_agent_executed)
        self.assertIsNone(self.service.event_store.latest_event(with_evidence=True))

    def test_tool_budget_is_enforced_even_if_agent_repeats_calls(self):
        self.reasoning.repeats = 4
        self.workflow.run('Research BTC')
        self.assertEqual(len(self.collector.calls), 1)

    def test_oversized_or_directionally_biased_queries_do_not_reach_tavily(self):
        for queries in (['Bitcoin news'] * 4, ['why Bitcoin will fall'], ['buy Bitcoin now']):
            self.reasoning.queries = queries
            self.workflow.run('Research BTC')
        self.assertEqual(self.collector.calls, [])

    def test_review_followup_request_is_recorded_without_extra_search(self):
        self.reasoning.more = True
        result = self.workflow.run('Research BTC')
        self.assertTrue(result.review.additional_research_needed)
        self.assertEqual(len(self.collector.calls), 1)

    def test_research_or_review_failure_withholds_evidence(self):
        for failed in ('Research', 'Review'):
            self.reasoning.fail = failed
            result = self.workflow.run('Research BTC')
            self.assertEqual(result.status, 'unavailable')
            self.assertEqual(result.evidence, [])
            self.assertTrue(result.workflow.errors)

    def test_all_internal_question_examples_do_not_search(self):
        for question in ('Explain the 6h forecast', 'What model generated the 24h forecast?',
                         'What does probability_up mean?', 'Why is reliability unavailable?',
                         'Compare the stored 1h and 24h forecasts'):
            self.assertFalse(needs_research(question))
            self.assertEqual(self.client.post('/chat', json={'message': question}).status_code, 200)
        self.assertEqual(self.collector.calls, [])
        self.assertEqual(self.reasoning.calls, [])

    def test_external_question_examples_route_to_research(self):
        for question in ('Why is BTC dropping right now?', 'Research the latest BTC news',
                         "What external events might explain today's move?"):
            self.assertTrue(needs_research(question))

    def checked(self, reliability=None, directions=(.1, .9, .1)):
        raw = copy.deepcopy(self.fixture.raw)
        for key, p in zip(('1', '6', '24'), directions):
            raw['forecasts'][key]['probability_up'] = p
            raw['forecasts'][key]['probability_down'] = 1-p
            raw['forecasts'][key]['validated_reliability'] = reliability
        context = self.workflow.forecast.run(raw)
        self.collector.result.review = ReviewReport(status='ok', accepted_evidence=['bull'])
        return self.workflow.validator.validate(context, raw, self.collector.result)

    def test_unknown_cannot_be_promoted_by_bullish_proposal_or_news(self):
        context, research, validation = self.checked()
        proposal = DecisionProposal(market_stance='BULLISH', decision_strength='HIGH', time_horizon='mixed',
            ml_view='BULLISH', research_view='BULLISH', agreement=True, short_term_risk='None',
            explanation='Reliable bullish forecast', evidence_ids=['invented'])
        decision = enforce_decision(context, research, validation, ['1h', '6h', '24h'], proposal)
        self.assertEqual(decision.market_stance, 'NO_RELIABLE_VIEW')
        self.assertEqual(decision.ml_view, 'NO_RELIABLE_SIGNAL')
        self.assertEqual(decision.research_view, 'BULLISH')
        self.assertEqual(decision.decision_strength, 'LOW')
        self.assertNotIn('invented', decision.evidence_ids)
        self.assertNotIn('Reliable bullish forecast', decision.explanation)

    def test_conflicting_qualified_horizons_produce_mixed(self):
        context, research, validation = self.checked(.8)
        decision = enforce_decision(context, research, validation, ['1h', '6h', '24h'])
        self.assertEqual(decision.ml_view, 'MIXED')
        self.assertEqual(decision.market_stance, 'MIXED')

    def test_qualified_longer_horizons_can_be_scoped_with_short_term_risk(self):
        context, research, validation = self.checked(.8, (.1, .9, .9))
        proposal = DecisionProposal(market_stance='BULLISH', decision_strength='HIGH', time_horizon='6-24h',
            ml_view='BULLISH', research_view='BULLISH', agreement=True, short_term_risk='None', explanation='', evidence_ids=[])
        decision = enforce_decision(context, research, validation, ['1h', '6h', '24h'], proposal)
        self.assertEqual(decision.market_stance, 'BULLISH')
        self.assertEqual(decision.time_horizon, '6-24h')
        self.assertIn('1h', decision.short_term_risk)
        self.assertIn('DOWN', decision.short_term_risk)
        one = enforce_decision(context, research, validation, ['1h'], proposal)
        self.assertEqual(one.time_horizon, '1h')
        self.assertEqual(one.ml_view, 'BEARISH')

    def test_workflow_audit_is_persisted_with_final_gate_and_stance(self):
        response = self.client.post('/research', json={}).json()
        stored = self.service.event_store.latest_event()['result']
        self.assertEqual(stored['workflow'], response['research']['workflow'])
        self.assertEqual(stored['workflow']['final_market_stance'], 'NO_RELIABLE_VIEW')
        self.assertEqual(set(stored['workflow']['final_signal_states'].values()), {'UNKNOWN'})
        self.assertEqual(self.client.get('/evidence').json()['market_decision']['market_stance'], 'NO_RELIABLE_VIEW')


class SDKRolesTests(unittest.TestCase):
    def test_actual_runner_executes_research_tool_and_review_with_mock_http(self):
        import httpx
        from openai import AsyncOpenAI
        from test_analyst_hardening import response_payload
        config = settings(openai_key='offline-key', openai_model='gpt-5.6')
        requests, searches = [], []
        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) == 1:
                response = response_payload({})
                response['output'] = [{'type': 'function_call', 'id': 'fc_offline', 'call_id': 'call_offline',
                    'name': 'search_tavily', 'arguments': json.dumps({'queries': ['Bitcoin macro news']}), 'status': 'completed'}]
            else:
                response = response_payload(dict(evidence_ids=['bull'], research_summary='One supplied report.'))
            return httpx.Response(200, json=response)
        def search(queries):
            searches.append(queries)
            return {'evidence': [article().model_dump()]}
        client = AsyncOpenAI(api_key='offline-key', http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)), max_retries=0)
        with patch('openai.AsyncOpenAI', return_value=client):
            output, status, executed = ReasoningAgents(config).run('Research',
                {'trigger_reason': ['price_move'], 'search_required': True}, search)
        self.assertEqual(status, 'ok')
        self.assertTrue(executed)
        self.assertEqual(output.evidence_ids, ['bull'])
        self.assertEqual(len(requests), 2)
        self.assertEqual(searches, [['Bitcoin macro news']])
        self.assertTrue(requests[0]['text']['format']['strict'])
        self.assertFalse(requests[0]['store'])
        self.assertEqual(requests[0]['tool_choice'], 'required')
        review = ReviewPlan(assessments=[EvidenceReview(evidence_id='bull', accept=False, stance='NEUTRAL', reason='Insufficient evidence')],
            contradictions=[], unsupported_claims=['Weak claim'], review_summary='Withheld.', additional_research_needed=False)
        review_requests = []
        def respond_review(request):
            review_requests.append(json.loads(request.content))
            return httpx.Response(200, json=response_payload(review.model_dump()))
        client = AsyncOpenAI(api_key='offline-key', http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond_review)), max_retries=0)
        with patch('openai.AsyncOpenAI', return_value=client):
            output, status, _ = ReasoningAgents(config).run('Review', {'evidence': [article().model_dump()]})
        self.assertEqual(status, 'ok')
        self.assertFalse(output.assessments[0].accept)
        self.assertFalse(review_requests[0].get('tools'))

    def test_three_actual_sdk_agent_definitions_have_separate_roles_and_tools(self):
        from agents import Agent, RunContextWrapper
        from agents.tool_context import ToolContext
        from agents.agent_output import AgentOutputSchema
        config = settings(openai_key='sk-offline-test', openai_model='gpt-5.6')
        calls, tool_calls = [], []
        async def run(agent, **kwargs):
            self.assertIsInstance(agent, Agent)
            AgentOutputSchema(agent.output_type)
            calls.append((agent.name, len(agent.tools), kwargs['max_turns']))
            if agent.name == 'BTC Research Agent':
                arguments = json.dumps({'queries': ['Bitcoin ETF flows']})
                tool_context = ToolContext.from_agent_context(RunContextWrapper(context=None), 'offline-call',
                    tool_name='search_tavily', tool_arguments=arguments, run_config=kwargs['run_config'])
                value = await agent.tools[0].on_invoke_tool(tool_context, arguments)
                self.assertIn('evidence', value)
                output = ResearchPlan(evidence_ids=[], research_summary='No sources available.')
            elif agent.name == 'BTC Review Agent':
                output = ReviewPlan(assessments=[], contradictions=[], unsupported_claims=[], review_summary='No sources.', additional_research_needed=False)
            else:
                output = DecisionPlan(focus_horizons=['1h'], fact_ids=[], evidence_ids=[], interpretation='insufficient_evidence', answer_style='concise')
            return SimpleNamespace(final_output=output)
        def search(queries):
            tool_calls.append(queries)
            return {'evidence': []}
        with patch('agents.Runner.run', side_effect=run):
            agents = ReasoningAgents(config)
            self.assertIsInstance(asyncio.run(agents._run('Research', {}, search)), ResearchPlan)
            self.assertIsInstance(asyncio.run(agents._run('Review', {})), ReviewPlan)
            asyncio.run(LLMService(config)._run_agent('{}'))
        self.assertEqual(calls, [('BTC Research Agent', 1, 2), ('BTC Review Agent', 0, 1), ('BTC Decision Agent', 0, 1)])
        self.assertEqual(tool_calls, [['Bitcoin ETF flows']])
