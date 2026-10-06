"""AgentCore Platform v1.0"""

# INS-C2-015 — PostProcessNode
# Outer backbone post_process slot: the output boundary. It screens the
# assembled FNOL response for credential material and, only if it is clean,
# exposes it as formatted_output and result.
#
# Two properties this node exists to hold:
#
#   1. The screen is a SUPERSET of the framework's own credential detector.
#      The framework re-scans every value of every node result and RAISES on a
#      match; a raise is caught by the node wrapper, which returns a bare error
#      partial and discards whatever this node decided. So a pattern the
#      framework knows and this node does not is not a smaller net — it is a
#      bypass of the containment below. detect_credentials() is called directly
#      here so the two sets can never drift apart.
#
#   2. On a violation the node CLEARS every output-bearing state field rather
#      than only marking the status. The framework's get_output() falls back to
#      state["result"] regardless of status, so leaving the ungated document in
#      state would ship it inside the error envelope.
#
# The domain screen is a module-level function (_security_gate_output) called
# from inside execute() — not an instance method on the node class, which the
# framework marks final and refuses to let a domain node override.
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from framework.security.credential_detector import detect_credentials
from shared.utils.audit_logger import emit_trace_event
from src.services.llm_factory import resolve_llm
from src.services.llm_review import render_review, review_result

logger = logging.getLogger(__name__)

# Domain patterns ON TOP OF the framework detector — never instead of it.
# A `password = ...` assignment is not a credential SHAPE the framework
# recognises, but it is exactly what a mis-pasted operations note looks like.
_EXTRA_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?:password|passwd|secret|api_key|token|access_key|private_key)" r"\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
        "credential_assignment",
    ),
)

# Every state field that can carry the ungated document. On a violation each
# one is cleared, so no representation of the answer survives in state for
# get_output() — or any later consumer — to fall back to.
_OUTPUT_BEARING_FIELDS = ("fnol_response", "grounded_answer", "reranked_docs")

_WITHHELD_NOTICE = (
    "[FNOL RESPONSE WITHHELD: the generated acknowledgment did not clear the "
    "output boundary and has not been released. Please contact claims support "
    "and quote this request's correlation id.]"
)

_EMPTY_RESPONSE_NOTICE = "[FNOL Response] No response content generated. " "Check error_log for upstream failures."


def _security_gate_output(value: Any) -> Optional[str]:
    """Scan *value* for credential material; return the finding type or None.

    Walks dicts, lists and tuples depth-first so a credential riding inside a
    nested structure is seen — a scan of top-level strings only would report
    zero findings on exactly the payload shape that carries one. Every string
    leaf is screened with the framework's detector first, then with the domain
    patterns above.
    """
    return _scan(value, 0)


def _scan(value: Any, depth: int) -> Optional[str]:
    if depth > 8:
        return None
    if isinstance(value, str):
        findings = detect_credentials(value)
        if findings:
            return str(findings[0]["type"])
        for pattern, name in _EXTRA_PATTERNS:
            if pattern.search(value):
                return name
        return None
    if isinstance(value, dict):
        for nested in value.values():
            found = _scan(nested, depth + 1)
            if found:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _scan(item, depth + 1)
            if found:
                return found
    return None


class PostProcessNode(FunctionNode):
    """Apply the output boundary and expose the final FNOL response.

    Outer backbone post_process slot.  Declared ANONYMOUS — trust was already
    enforced at PreProcessNode (VERIFIED_EXTERNAL).

    Input state keys:
        fnol_response: str  — formatted response from inner OutputFormatNode

    Output state keys (partial dict):
        formatted_output: str
        result:           str
        status:           str
        error_log:        list[str]  (only on ERROR)
        fnol_response / grounded_answer / reranked_docs — cleared on a violation
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        fnol_response: str = state.get("fnol_response") or state.get("result") or ""

        # ── Fallback for empty response ───────────────────────────────────────
        if not fnol_response.strip():
            logger.warning("PostProcessNode: fnol_response is empty — using fallback message")
            fnol_response = _EMPTY_RESPONSE_NOTICE

        # ── Output boundary ───────────────────────────────────────────────────
        # Screen the document AND every other output-bearing field: the
        # rendered text is the caller-visible representation, but the same
        # material can ride in the structured intermediates, and those are
        # what the framework's own re-scan sees.
        _llm, _ = resolve_llm(None, state)
        _remarks = review_result(
            _llm,
            user_input=str(state.get("user_input") or ""),
            result=fnol_response,
            domain="INS FNOLProcessingAgent",
        )
        _review = render_review(_remarks)
        # Remarks are LLM text derived from the caller's raw words, so they pass through the
        # same gate the answer does -- appending after the gate would put unscanned text past
        # it. A tripped review is dropped on its own: withholding a correct answer because an
        # advisory remark quoted an identifier would let the review change the outcome, and
        # the whole design rests on it being unable to.
        if (
            _review
            and isinstance(fnol_response, str)
            and not _security_gate_output({"fnol_response": fnol_response + _review})
        ):
            fnol_response = fnol_response + _review

        candidate: dict[str, Any] = {"fnol_response": fnol_response}
        for field in _OUTPUT_BEARING_FIELDS:
            if field != "fnol_response":
                candidate[field] = state.get(field)

        violation = _security_gate_output(candidate)
        if violation:
            logger.error("PostProcessNode: output boundary violation — %s", violation)
            emit_trace_event(
                "post_process_output_violation",
                {"violation": violation},
                state,
            )
            withheld: dict[str, Any] = {
                "formatted_output": _WITHHELD_NOTICE,
                "result": _WITHHELD_NOTICE,
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PostProcessNode: output withheld — credential pattern detected ({violation})"],
            }
            # Clear every field that could still carry the ungated document.
            for field in _OUTPUT_BEARING_FIELDS:
                withheld[field] = None
            return withheld

        logger.info("PostProcessNode: output boundary passed — length=%d", len(fnol_response))
        emit_trace_event(
            "post_process_complete",
            {"output_length": len(fnol_response)},
            state,
        )

        return {
            "formatted_output": fnol_response,
            "result": fnol_response,
            "status": AgentStatus.SUCCESS.value,
        }
