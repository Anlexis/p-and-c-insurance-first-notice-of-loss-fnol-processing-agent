# INS-C2-015 — Unit Tests: the flat-composition reference node + the trust gate

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from src.nodes.main_node import MainNode
from src.nodes.pre_process_node import PreProcessNode


class TestMainNode:
    """Unit tests for the flat-composition reference node.

    Node instances are invoked as ``node(state)`` (i.e. through
    ``BaseNode.__call__``) — NOT via a direct ``node.execute(state)`` call —
    so the mandatory trust gate and the input/output gates run on every
    invocation exactly as they do in a deployed agent.
    """

    def setup_method(self):
        self.node = MainNode()

    def test_success_path(self):
        """The reference node returns SUCCESS and a non-empty result."""
        state = {
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "validated_input": "test input",
            "node_history": [],
            "error_log": [],
        }
        result = self.node(state)  # __call__ → trust gate → execute()
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["result"]

    def test_empty_input(self):
        """The reference node handles empty input gracefully."""
        state = {
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "validated_input": "",
            "node_history": [],
            "error_log": [],
        }
        result = self.node(state)  # __call__ → trust gate → execute()
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_execute_method_signature(self):
        """Node contract: implement execute(state), never a private invoke hook."""
        import inspect

        assert hasattr(MainNode, "execute"), "MainNode must implement execute()"

        sig = inspect.signature(MainNode.execute)
        params = list(sig.parameters.keys())
        assert len(params) >= 2, f"execute() must accept (self, state), got params: {params}"
        assert params[1] == "state", f"Second parameter must be 'state', got '{params[1]}'"

        assert (
            "_invoke_impl" not in MainNode.__dict__
        ), "_invoke_impl() must not be defined in MainNode — use execute() instead"


class TestTrustGate:
    """Trust gate — negative authorization + positive control.

    PreProcessNode is the outer backbone's only VERIFIED_EXTERNAL node
    (``required_trust_level = TrustLevel.VERIFIED_EXTERNAL``); every other node
    is ANONYMOUS and can never deny.  These tests invoke it through
    ``node(state)`` (BaseNode.__call__) so the gate runs *before* execute():

      - an ANONYMOUS caller is REJECTED before execute() ever runs, and
      - a VERIFIED_EXTERNAL caller is admitted and processed.

    ``BaseNode.__call__`` returns an error dict on denial — it does NOT raise —
    so the assertions are on the returned dict, not on pytest.raises.
    """

    def setup_method(self):
        self.node = PreProcessNode()

    def test_required_trust_level_is_verified_external(self):
        assert self.node.required_trust_level == TrustLevel.VERIFIED_EXTERNAL

    def test_trust_gate_rejects_untrusted_caller_before_execute(self):
        state = {
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "user_input": '{"claim_id": "CLAIM-1", "query": "coverage question"}',
            "input_context": {},
            "node_history": [],
            "error_log": [],
        }
        result = self.node(state)  # __call__, NOT .execute() — the gate runs first
        assert result["status"] == AgentStatus.ERROR.value
        assert any("trust gate denied" in e for e in result["error_log"])
        # execute() never ran → its output key is absent (short-circuited).
        assert "validated_input" not in result

    def test_trust_gate_admits_trusted_caller(self):
        state = {
            "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
            "user_input": '{"claim_id": "CLAIM-1", "query": "coverage question"}',
            "input_context": {},
            "node_history": [],
            "error_log": [],
        }
        result = self.node(state)  # __call__ → gate admits → execute() runs
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] is not None
