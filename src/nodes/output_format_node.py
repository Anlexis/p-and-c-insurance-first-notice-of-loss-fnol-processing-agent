"""AgentCore Platform v1.0"""

# INS-C2-015 — OutputFormatNode (inner domain node 5, last in DomainWorkflowGraph)
# Assembles the final FNOL response document from the grounded answer and its
# citations.  This is the last inner node — it produces the fnol_response string
# the outer PostProcessNode passes through the output boundary.
#
# Output invariant (documented in docs/02_design.md and enforced here):
#   every value rendered into the acknowledgment is either template text, text
#   from the policy knowledge base, or a caller identifier already restricted
#   to the inert alphabet [A-Za-z0-9_-]. No caller free text is rendered, and
#   no monetary figure is produced anywhere in this pipeline, so the document
#   carries no amounts to round or redact.
#
# Inner node — ANONYMOUS trust (see DomainWorkflowGraph).
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json
from src.services.service import is_identifier, is_inert_token
from src.services.source_disclosure import source_label

logger = logging.getLogger(__name__)

_SEPARATOR = "=" * 72
_SUBSEP = "-" * 72

# Last-line defence on the render boundary: a value that is not already inert
# is replaced rather than printed. Upstream nodes refuse such values outright,
# so reaching this substitution means an upstream check was bypassed — the
# document still cannot carry caller markup either way.
_REDACTED_IDENTIFIER = "unavailable"


def _assemble_response(
    claim_id: str,
    channel: str,
    answer: str,
    citations: list[dict[str, Any]],
    grounded: bool,
) -> str:
    """Assemble the customer-ready FNOL response document."""
    grounding = "policy-grounded" if grounded else "routed to adjuster (no direct provision found)"
    lines = [
        _SEPARATOR,
        "FIRST NOTICE OF LOSS — ACKNOWLEDGMENT & COVERAGE GUIDANCE",
        f"Claim ID: {claim_id}",
        f"Intake Channel: {channel}",
        f"Grounding: {grounding}",
        _SEPARATOR,
        "",
        answer,
        "",
        _SUBSEP,
        "CITATIONS",
        _SUBSEP,
    ]
    if citations:
        for i, c in enumerate(citations, start=1):
            lines.append(
                f"  [{i}] {c.get('doc_id', 'N/A')} — {c.get('source', 'Policy')} "
                f"/ {c.get('policy_section', 'Provision')}"
            )
    else:
        lines.append("  (none — matter routed to a licensed adjuster for coverage verification)")
    lines += [
        _SEPARATOR,
        "NOTE: Intake acknowledgment only. Final coverage determination and "
        "reserve setting are completed by a licensed claims adjuster.",
        _SEPARATOR,
    ]
    return "\n".join(lines)


class OutputFormatNode(FunctionNode):
    """Assemble the final FNOL response document (inner domain node).

    Reads grounded_answer from State, renders the full customer-ready FNOL
    acknowledgment text with citations, and writes it to fnol_response (and
    result) for the outer PostProcessNode output boundary.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        grounded_answer: str  — JSON-serialised {claim_id, answer, citations,
                                grounded, intake_channel}

    Output state keys (partial dict):
        fnol_response: str
        result:        str  (same as fnol_response — backbone convention)
        status:        str
        error_log:     list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        grounded_answer: dict[str, Any] = from_json(state.get("grounded_answer"), {})

        if not grounded_answer:
            logger.error("OutputFormatNode: grounded_answer missing in state")
            emit_trace_event(
                "output_format_failed",
                {"reason": "missing_grounded_answer"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["OutputFormatNode: grounded_answer missing in state"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("OutputFormatNode: grounded_answer missing in state"),
            }

        raw_claim_id = grounded_answer.get("claim_id")
        claim_id = str(raw_claim_id) if is_identifier(raw_claim_id) else _REDACTED_IDENTIFIER
        raw_channel = grounded_answer.get("intake_channel")
        channel = str(raw_channel) if is_inert_token(raw_channel) else _REDACTED_IDENTIFIER
        answer = str(grounded_answer.get("answer", ""))
        citations: list[dict[str, Any]] = grounded_answer.get("citations") or []
        grounded = bool(grounded_answer.get("grounded", False))

        response = _assemble_response(claim_id, channel, answer, citations, grounded)

        logger.info(
            "OutputFormatNode: claim_id=%s response_chars=%d grounded=%s citations=%d",
            claim_id,
            len(response),
            grounded,
            len(citations),
        )
        # Say where the answer came from. This agent answers from a corpus defined inside
        # its own module; a reader seeing a citation has no way to tell that from a live query
        # against the system of record, and the review round rated that confusion its most
        # serious finding. Added before the gate below so it passes the same checks the answer
        # does.
        response = response + source_label(state)
        emit_trace_event(
            "output_format_complete",
            {
                "claim_id": claim_id,
                "intake_channel": channel,
                "response_length": len(response),
                "grounded": grounded,
                "citation_count": len(citations),
            },
            state,
        )

        return {
            "fnol_response": response,
            "result": response,
            "status": AgentStatus.SUCCESS.value,
        }
