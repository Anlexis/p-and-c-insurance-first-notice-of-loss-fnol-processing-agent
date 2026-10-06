# INS-C2-015 — Unit Tests: domain nodes + graph wiring
#
# Real, non-stub unit tests. They import the real modules and assert real
# behaviour (RAG retrieval scoring, grounded vs. abstained answers, trust
# levels, the output boundary, and the Cat 2 two-layer nested composition).
#
# INS-C2-015 is a Cat 2 RAG agent (P&C insurance FNOL processing): outer
# `AgentBaseGraph` backbone + inner `DomainWorkflowGraph` (`BaseGraph`) running
# a linear RAG pipeline
#   input_validate -> retrieve -> rerank_filter -> generate_answer -> output_format.
#
# Audit events are patched at the node MODULE level (not via a sys.modules
# stub, which would break the real `shared` package the framework loads at import
# time). Patch pattern per node:
#     monkeypatch.setattr("src.nodes.<mod>.emit_trace_event", lambda *a, **k: None)

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from src.schemas.state import from_json, to_json


# ── Shared fixtures / helpers ─────────────────────────────────────────────────


def _claim_payload(**overrides) -> dict:
    """A complete, valid raw FNOL claim-query payload (as a caller would POST).

    The default query is an auto-glass windshield question that lexically
    overlaps the bundled comprehensive/glass policy passages, so the RAG
    pipeline retrieves and grounds an answer (grounded=True).
    """
    payload = {
        "claim_id": "INS-FNOL-20260712-001",
        "claim_type": "auto",
        "policy_number": "PC-AUTO-88231",
        "query": (
            "My windshield was cracked by road debris — is the glass damage "
            "covered under my comprehensive auto policy, and does the "
            "deductible apply?"
        ),
    }
    payload.update(overrides)
    return payload


VALID_PAYLOAD = json.dumps(_claim_payload())

# A query with no lexical overlap with any policy passage — drives abstention
# (retrieved passages carry only the claim-type bonus 0.10 < score_threshold
# 0.15, so none survive the rerank filter → grounded=False).
ABSTAIN_PAYLOAD = json.dumps(
    _claim_payload(
        claim_type="unknown",
        query="What is the capital of France and its tourism season?",
    )
)


def _validated_query(**overrides) -> dict:
    """The normalised validated_query object shape produced by InputValidateNode
    (i.e. the input the downstream inner RAG nodes consume)."""
    q = {
        "claim_id": "INS-FNOL-20260712-001",
        "claim_type": "auto",
        "query": ("windshield glass cracked by road debris — comprehensive coverage " "and deductible"),
        "policy_number": "PC-AUTO-88231",
    }
    q.update(overrides)
    return q


# ── PreProcessNode (outer pre_process, VERIFIED_EXTERNAL) ─────────────────────


class TestPreProcessNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.pre_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.pre_process_node import PreProcessNode

        self.node = PreProcessNode()

    def test_valid_payload_returns_success(self):
        result = self.node.execute({"user_input": VALID_PAYLOAD, "input_context": {}})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] is not None
        assert json.loads(result["validated_input"])["claim_id"] == "INS-FNOL-20260712-001"

    def test_enriched_context_carries_claim_id(self):
        result = self.node.execute({"user_input": VALID_PAYLOAD, "input_context": {"channel": "web"}})
        ctx = from_json(result["enriched_context"])
        assert ctx["claim_id"] == "INS-FNOL-20260712-001"
        assert ctx["channel"] == "web"
        assert ctx["source"] == "FNOLProcessingAgent"

    def test_empty_input_returns_error(self):
        result = self.node.execute({"user_input": "", "input_context": {}})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("empty" in e for e in result["error_log"])

    def test_invalid_json_returns_error(self):
        result = self.node.execute({"user_input": "{not valid json}", "input_context": {}})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("JSON" in e or "json" in e for e in result["error_log"])

    def test_non_object_json_returns_error(self):
        result = self.node.execute({"user_input": "[1, 2, 3]", "input_context": {}})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("object" in e for e in result["error_log"])

    def test_missing_required_field_returns_error(self):
        payload = {"claim_id": "X"}  # no 'query'
        result = self.node.execute({"user_input": json.dumps(payload), "input_context": {}})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("query" in e for e in result["error_log"])

    def test_empty_query_returns_error(self):
        payload = _claim_payload(query="   ")
        result = self.node.execute({"user_input": json.dumps(payload), "input_context": {}})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("query" in e for e in result["error_log"])

    def test_trust_level_is_verified_external(self):
        assert self.node.required_trust_level == TrustLevel.VERIFIED_EXTERNAL

    def test_execute_signature_is_state_first(self):
        import inspect
        from src.nodes.pre_process_node import PreProcessNode

        params = list(inspect.signature(PreProcessNode.execute).parameters.keys())
        assert params[0] == "self" and params[1] == "state"
        assert "_invoke_impl" not in PreProcessNode.__dict__


# ── InputValidateNode (inner domain node 1, ANONYMOUS) ─────────────────────────


class TestInputValidateNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.input_validate_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.input_validate_node import InputValidateNode

        self.node = InputValidateNode()

    def test_valid_input_builds_validated_query(self):
        result = self.node.execute({"validated_input": VALID_PAYLOAD})
        assert result["status"] == AgentStatus.SUCCESS.value
        q = from_json(result["validated_query"])
        assert q["claim_id"] == "INS-FNOL-20260712-001"
        assert q["claim_type"] == "auto"
        assert q["policy_number"] == "PC-AUTO-88231"
        assert q["query"]

    def test_claim_type_aliases_are_normalised(self):
        for alias, canonical in (("car", "auto"), ("home", "fire"), ("injury", "accident")):
            payload = _claim_payload(claim_type=alias)
            q = from_json(self.node.execute({"validated_input": json.dumps(payload)})["validated_query"])
            assert q["claim_type"] == canonical, f"{alias} should normalise to {canonical}"

    def test_unknown_claim_type_falls_back_to_unknown(self):
        payload = _claim_payload(claim_type="spaceship")
        q = from_json(self.node.execute({"validated_input": json.dumps(payload)})["validated_query"])
        assert q["claim_type"] == "unknown"

    def test_falls_back_to_user_input(self):
        result = self.node.execute({"user_input": VALID_PAYLOAD})
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_long_query_is_truncated(self):
        payload = _claim_payload(query="glass " * 1000)  # >2000 chars
        q = from_json(self.node.execute({"validated_input": json.dumps(payload)})["validated_query"])
        assert len(q["query"]) <= 2000

    def test_empty_query_returns_error(self):
        payload = _claim_payload(query="")
        result = self.node.execute({"validated_input": json.dumps(payload)})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("query" in e for e in result["error_log"])

    def test_non_object_payload_returns_error(self):
        result = self.node.execute({"validated_input": "[1, 2, 3]"})
        assert result["status"] == AgentStatus.ERROR.value

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── RetrieveNode (inner domain node 2 — RAG retrieve, ANONYMOUS) ───────────────


class TestRetrieveNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.retrieve_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.retrieve_node import RetrieveNode

        self.node = RetrieveNode()

    def test_retrieves_relevant_passages(self):
        result = self.node.execute({"validated_query": to_json(_validated_query())})
        assert result["status"] == AgentStatus.SUCCESS.value
        docs = from_json(result["retrieved_docs"])
        assert len(docs) > 0
        # Each retrieved passage carries the stable contract shape.
        for d in docs:
            assert set(d.keys()) >= {"doc_id", "text", "score", "source", "policy_section"}
        # The auto-glass windshield question must surface the comprehensive passage.
        assert any(d["doc_id"] == "AUTO-COMP-001" for d in docs)

    def test_scores_sorted_descending(self):
        docs = from_json(self.node.execute({"validated_query": to_json(_validated_query())})["retrieved_docs"])
        scores = [d["score"] for d in docs]
        assert scores == sorted(scores, reverse=True)

    def test_declared_top_k_limits_results(self):
        """The declared top_k reaches the node through runtime_settings."""
        state = {
            "validated_query": to_json(_validated_query()),
            "runtime_settings": to_json({"top_k": 2}),
        }
        docs = from_json(self.node.execute(state)["retrieved_docs"])
        assert len(docs) <= 2

    def test_non_finite_top_k_falls_back_to_the_default(self):
        """A non-finite or out-of-range declaration must not disable retrieval."""
        for bad in (float("nan"), float("inf"), -1, 0, 10**9, True, "5", None):
            state = {
                "validated_query": to_json(_validated_query()),
                "runtime_settings": json.dumps({"top_k": bad}),
            }
            docs = from_json(self.node.execute(state)["retrieved_docs"])
            assert 0 < len(docs) <= 5, f"top_k={bad!r} must fall back to the default of 5"

    def test_missing_query_returns_error(self):
        result = self.node.execute({"validated_query": to_json({"query": "", "claim_type": "auto"})})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("validated_query" in e for e in result["error_log"])

    def test_does_not_mutate_module_kb(self):
        from src.nodes import retrieve_node

        before = json.dumps(retrieve_node._POLICY_KB, sort_keys=True)
        self.node.execute({"validated_query": to_json(_validated_query())})
        after = json.dumps(retrieve_node._POLICY_KB, sort_keys=True)
        assert before == after, "RetrieveNode must not mutate the module-global _POLICY_KB"

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── RerankFilterNode (inner domain node 3 — RAG rerank/filter, ANONYMOUS) ──────


class TestRerankFilterNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.rerank_filter_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.rerank_filter_node import RerankFilterNode

        self.node = RerankFilterNode()

    def _docs(self):
        return [
            {"doc_id": "A", "text": "aaaa", "score": 0.90, "source": "S", "policy_section": "P"},
            {"doc_id": "B", "text": "bbbbbbbb", "score": 0.20, "source": "S", "policy_section": "P"},
            {"doc_id": "C", "text": "cccc", "score": 0.05, "source": "S", "policy_section": "P"},
        ]

    def test_filters_below_threshold(self):
        result = self.node.execute({"retrieved_docs": to_json(self._docs())})
        kept = from_json(result["reranked_docs"])
        # default threshold 0.15 → C (0.05) dropped, A and B kept.
        assert {d["doc_id"] for d in kept} == {"A", "B"}
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_sorts_by_score_descending(self):
        kept = from_json(self.node.execute({"retrieved_docs": to_json(self._docs())})["reranked_docs"])
        assert [d["doc_id"] for d in kept] == ["A", "B"]

    def test_declared_threshold_reaches_the_node(self):
        """The declared score_threshold reaches the node through runtime_settings."""
        state = {
            "retrieved_docs": to_json(self._docs()),
            "runtime_settings": to_json({"score_threshold": 0.5}),
        }
        kept = from_json(self.node.execute(state)["reranked_docs"])
        assert {d["doc_id"] for d in kept} == {"A"}

    def test_non_finite_threshold_falls_back_to_the_default(self):
        """NaN compares False against every bound — it must never become the bar."""
        for bad in (float("nan"), float("inf"), float("-inf"), -1.0, 2.0, True, "0.5", None):
            state = {
                "retrieved_docs": to_json(self._docs()),
                "runtime_settings": json.dumps({"score_threshold": bad}),
            }
            kept = from_json(self.node.execute(state)["reranked_docs"])
            assert {d["doc_id"] for d in kept} == {
                "A",
                "B",
            }, f"score_threshold={bad!r} must fall back to the default of 0.15"

    def test_passage_with_a_non_finite_score_is_dropped(self):
        """An unscored passage must not be able to ground an answer."""
        docs = self._docs() + [
            {"doc_id": "D", "text": "dddd", "score": float("nan"), "source": "S", "policy_section": "P"},
            {"doc_id": "E", "text": "eeee", "score": "0.99", "source": "S", "policy_section": "P"},
        ]
        state = {"retrieved_docs": json.dumps(docs)}
        kept = from_json(self.node.execute(state)["reranked_docs"])
        assert {d["doc_id"] for d in kept} == {"A", "B"}

    def test_empty_input_is_success_with_empty_list(self):
        result = self.node.execute({"retrieved_docs": to_json([])})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert from_json(result["reranked_docs"]) == []

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── GenerateAnswerNode (inner domain node 4 — grounded generation, ANONYMOUS) ──


class TestGenerateAnswerNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.generate_answer_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.generate_answer_node import GenerateAnswerNode

        self.node = GenerateAnswerNode()

    def _reranked(self):
        return [
            {
                "doc_id": "AUTO-COMP-001",
                "text": "Comprehensive coverage pays for glass breakage.",
                "score": 0.9,
                "source": "Comprehensive Auto Policy",
                "policy_section": "Section III",
            },
        ]

    def test_grounded_answer_cites_passages(self):
        state = {
            "reranked_docs": to_json(self._reranked()),
            "validated_query": to_json(_validated_query()),
        }
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        ga = from_json(result["grounded_answer"])
        assert ga["grounded"] is True
        assert len(ga["citations"]) == 1
        assert ga["citations"][0]["doc_id"] == "AUTO-COMP-001"
        assert "First Notice of Loss" in ga["answer"]

    def test_abstains_when_no_passages(self):
        """No reranked passage → ungrounded abstention (never fabricate coverage)."""
        state = {
            "reranked_docs": to_json([]),
            "validated_query": to_json(_validated_query()),
        }
        result = self.node.execute(state)
        ga = from_json(result["grounded_answer"])
        assert ga["grounded"] is False
        assert ga["citations"] == []
        assert "adjuster" in ga["answer"].lower()

    def test_grounded_answer_carries_claim_id(self):
        state = {
            "reranked_docs": to_json(self._reranked()),
            "validated_query": to_json(_validated_query(claim_id="INS-XYZ")),
        }
        ga = from_json(self.node.execute(state)["grounded_answer"])
        assert ga["claim_id"] == "INS-XYZ"

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── OutputFormatNode (inner domain node 5, ANONYMOUS) ──────────────────────────


class TestOutputFormatNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.output_format_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.output_format_node import OutputFormatNode

        self.node = OutputFormatNode()

    def _grounded_answer(self, grounded=True):
        return {
            "claim_id": "INS-FNOL-20260712-001",
            "answer": "Thank you for submitting your First Notice of Loss.",
            "citations": (
                [{"doc_id": "AUTO-COMP-001", "source": "Comprehensive Auto Policy", "policy_section": "Section III"}]
                if grounded
                else []
            ),
            "grounded": grounded,
        }

    def test_assembles_full_response(self):
        result = self.node.execute({"grounded_answer": to_json(self._grounded_answer())})
        assert result["status"] == AgentStatus.SUCCESS.value
        response = result["fnol_response"]
        assert result["result"] == response
        assert "FIRST NOTICE OF LOSS" in response
        assert "INS-FNOL-20260712-001" in response
        assert "CITATIONS" in response
        assert "AUTO-COMP-001" in response
        assert "policy-grounded" in response

    def test_abstention_response_routes_to_adjuster(self):
        response = self.node.execute({"grounded_answer": to_json(self._grounded_answer(grounded=False))})[
            "fnol_response"
        ]
        assert "routed to adjuster" in response
        assert "(none" in response  # no citations block

    def test_missing_grounded_answer_returns_error(self):
        result = self.node.execute({})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("grounded_answer" in e for e in result["error_log"])

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── PostProcessNode (outer post_process, output boundary, ANONYMOUS) ──────────


class TestPostProcessNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.post_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.post_process_node import PostProcessNode

        self.node = PostProcessNode()

    def test_clean_response_passes_gate(self):
        response = "FIRST NOTICE OF LOSS — ACKNOWLEDGMENT\nClaim ID: INS-1\nCoverage guidance."
        result = self.node.execute({"fnol_response": response})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == response
        assert result["result"] == response

    def test_empty_response_uses_fallback(self):
        result = self.node.execute({"fnol_response": ""})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "No response content generated" in result["formatted_output"]

    def test_output_boundary_withholds_a_credential_leak(self):
        leaky = "FIRST NOTICE OF LOSS\ntoken=sk-abcdefghij0123456789ABCDEF"
        result = self.node.execute({"fnol_response": leaky})
        assert result["status"] == AgentStatus.ERROR.value
        assert "WITHHELD" in result["formatted_output"]
        assert result["formatted_output"] == result["result"]
        assert "sk-abcdefghij0123456789ABCDEF" not in result["formatted_output"]
        assert any("output withheld" in e for e in result["error_log"])

    def test_output_boundary_clears_every_output_bearing_field(self):
        """Marking the status is not containment — the document must be cleared.

        The framework's get_output() falls back to state["result"] regardless of
        status, so an ungated representation left in state ships inside the
        error envelope.
        """
        leaky = "FIRST NOTICE OF LOSS\ntoken=sk-abcdefghij0123456789ABCDEF"
        result = self.node.execute(
            {
                "fnol_response": leaky,
                "grounded_answer": to_json({"answer": leaky}),
                "reranked_docs": to_json([{"text": leaky}]),
            }
        )
        for field in ("fnol_response", "grounded_answer", "reranked_docs"):
            assert result[field] is None, f"{field} must be cleared on a violation"

    def test_output_boundary_sees_a_credential_nested_below_the_top_level(self):
        """A scan of top-level strings only reports zero findings here."""
        result = self.node.execute(
            {
                "fnol_response": "A perfectly clean FNOL acknowledgment.",
                "reranked_docs": to_json([{"text": "note: Bearer abcdefghijklmnop0123456789"}]),
            }
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert "WITHHELD" in result["formatted_output"]

    def test_output_boundary_covers_the_framework_detector(self):
        """The domain screen is a SUPERSET of the framework's own detector.

        A type the framework catches and this node misses is not a smaller net:
        the framework re-scans the result and RAISES, the node wrapper turns
        that into a bare error partial, and the clearing above is discarded.
        """
        from framework.security.credential_detector import detect_credentials
        from src.nodes.post_process_node import _security_gate_output

        framework_shapes = [
            "sk_live_" + "abcdefghijklmnop0123",
            "sk-abcdefghij0123456789ABCDEF",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefg",
            "AKIAIOSFODNN7EXAMPLE",
            "Authorization: Bearer abcdefghijklmnop0123456789",
            "postgresql://user:example-password@db.invalid:5432/claims",
        ]
        for shape in framework_shapes:
            assert detect_credentials(shape), f"fixture is not a framework match: {shape}"
            assert (
                _security_gate_output(shape) is not None
            ), f"the domain screen must catch everything the framework catches: {shape}"
        # Domain patterns on top of the framework set.
        assert _security_gate_output("password = supersecret123") == "credential_assignment"
        # And ordinary domain text is untouched.
        assert _security_gate_output("A perfectly clean FNOL acknowledgment.") is None
        assert _security_gate_output({"nested": ["clause AUTO-COMP-001 applies"]}) is None

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── Graph wiring: outer AgentBaseGraph + inner BaseGraph (Cat 2 nested) ─────────


class TestOuterGraphComposition:
    def test_registers_five_backbone_slots(self):
        from src.graph.graph import FNOLProcessingAgent, FNOLRetrievalGraphNode
        from src.nodes.post_process_node import PostProcessNode
        from src.nodes.pre_process_node import PreProcessNode

        agent = FNOLProcessingAgent()
        agent.compile()
        assert set(agent._nodes.keys()) == {
            "initialize",
            "pre_process",
            "main",
            "post_process",
            "finalize",
        }
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["main"], FNOLRetrievalGraphNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)

    def test_name_and_state_schema(self):
        from src.schemas.state import State
        from src.graph.graph import FNOLProcessingAgent

        agent = FNOLProcessingAgent()
        assert agent.name == "FNOLProcessingAgent"
        assert agent.state_schema is State

    def test_graph_alias_matches_real_class(self):
        from src.graph.graph import Graph, FNOLProcessingAgent

        assert Graph is FNOLProcessingAgent

    def test_main_slot_graphnode_contracts(self):
        from framework.nodes.graph_node import GraphNode
        from src.graph.graph import FNOLRetrievalGraphNode

        node = FNOLRetrievalGraphNode()
        assert isinstance(node, GraphNode)
        assert node.error_strategy == "propagate"
        assert node.propagate_hitl is False
        # extract_input prefers validated_input, falls back to user_input.
        assert node.extract_input({"validated_input": "V", "user_input": "U"}) == "V"
        assert node.extract_input({"user_input": "U"}) == "U"

    def test_merge_output_maps_subresult_keys(self):
        from src.graph.graph import FNOLRetrievalGraphNode

        node = FNOLRetrievalGraphNode()
        sub_result = {
            "fnol_response": "REPORT",
            "grounded_answer": "{}",
            "reranked_docs": "[]",
            "status": AgentStatus.SUCCESS.value,
            "node_history": ["x"],  # not forwarded by merge_output
            "correlation_id": "c",  # not forwarded by merge_output
        }
        delta = node.merge_output({}, sub_result)
        assert delta["fnol_response"] == "REPORT"
        assert delta["status"] == AgentStatus.SUCCESS.value
        assert set(delta.keys()) == {
            "fnol_response",
            "grounded_answer",
            "reranked_docs",
            "status",
        }


class TestInnerDomainGraph:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        for mod in (
            "input_validate_node",
            "retrieve_node",
            "rerank_filter_node",
            "generate_answer_node",
            "output_format_node",
        ):
            monkeypatch.setattr(f"src.nodes.{mod}.emit_trace_event", lambda *a, **k: None)

    def test_registers_five_domain_nodes(self):
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        g.register_nodes()
        assert set(g._nodes.keys()) == {
            "input_validate",
            "retrieve",
            "rerank_filter",
            "generate_answer",
            "output_format",
        }

    def test_name_and_state_schema(self):
        from src.schemas.state import State
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        assert g.name == "ins_c2_015_fnol_coverage_rag_workflow"
        assert g.state_schema is State

    def test_inner_graph_invoke_grounds_answer(self):
        """Standalone inner-graph invoke (ANONYMOUS caller) runs the linear RAG
        pipeline and shapes the get_output() dict consumed by outer merge_output()."""
        from framework.schemas.invocation_context import InvocationContext, TrustLevel
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        g.compile()
        ctx = InvocationContext(caller_trust_level=TrustLevel.ANONYMOUS)
        result = g.invoke(VALID_PAYLOAD, ctx=ctx)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["fnol_response"] is not None
        assert "FIRST NOTICE OF LOSS" in result["fnol_response"]
        assert len(from_json(result["reranked_docs"], [])) > 0

    def test_inner_graph_invoke_abstains_on_irrelevant_query(self):
        """A query with no policy-passage overlap must abstain (grounded=False):
        the RAG pipeline routes to an adjuster rather than fabricating coverage."""
        from framework.schemas.invocation_context import InvocationContext, TrustLevel
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        g.compile()
        ctx = InvocationContext(caller_trust_level=TrustLevel.ANONYMOUS)
        result = g.invoke(ABSTAIN_PAYLOAD, ctx=ctx)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert from_json(result["reranked_docs"], []) == []
        ga = from_json(result["grounded_answer"], {})
        assert ga.get("grounded") is False
