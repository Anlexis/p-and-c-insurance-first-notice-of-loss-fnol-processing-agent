"""AgentCore Platform v1.0"""

# INS-C2-015 — GenerateAnswerNode
# Inner domain node 4 (RAG grounded generation): produce a grounded FNOL
# answer / acknowledgment strictly from the reranked policy passages.
#
# The shipped agent is deterministic: the answer is assembled ONLY from the
# retrieved passage text, with inline citations, and is marked ungrounded when
# no passage survives the score_threshold (never fabricate coverage). The
# declared system-prompt template (prompts/fnol_coverage.j2) states the same
# grounding contract for a model-backed deployment of this pipeline; it is
# resolved and recorded here so the declared value is verified rather than
# assumed, and so the audit trail names the contract the answer was produced
# under.
#
# Inner node — ANONYMOUS trust (see DomainWorkflowGraph).
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from pathlib import Path
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json, to_json

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_TEMPLATE = "prompts/fnol_coverage.j2"

# src/nodes/generate_answer_node.py -> parents[2] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

_UNGROUNDED_ANSWER = (
    "We have received your First Notice of Loss. We could not locate a policy "
    "provision that directly addresses your question from the available policy "
    "knowledge base, so this matter has been routed to a licensed claims adjuster "
    "for coverage verification. No coverage determination is made at intake."
)


def _resolve_prompt_template(declared: Any) -> tuple[str, bool]:
    """Return the grounding-contract template path and whether it resolves.

    A declared path that does not exist in the deployed tree is reported as
    unresolved rather than silently accepted: the value would otherwise look
    live in the manifest while pointing at nothing.
    """
    template = declared if isinstance(declared, str) and declared else _DEFAULT_PROMPT_TEMPLATE
    return template, (_REPO_ROOT / template).is_file()


def _synthesise_grounded_answer(docs: list[dict[str, Any]]) -> str:
    """Build a grounded answer strictly from the reranked passages.

    The caller's question is deliberately NOT echoed into the answer: every
    sentence here originates either in this module or in the policy knowledge
    base, so caller text cannot reach the response document through this node.
    """
    lines: list[str] = [
        "Thank you for submitting your First Notice of Loss. Based on the "
        "applicable policy provisions, the following coverage guidance applies:",
        "",
    ]
    for i, doc in enumerate(docs, start=1):
        section = doc.get("policy_section", "Policy Provision")
        source = doc.get("source", "Policy")
        text = str(doc.get("text", "")).strip()
        lines.append(f"[{i}] ({source} — {section}) {text}")
    lines.append("")
    lines.append(
        "This acknowledgment is grounded solely in the cited provisions and is "
        "not a final coverage determination; a licensed adjuster completes "
        "coverage verification and reserve setting."
    )
    return "\n".join(lines)


class GenerateAnswerNode(FunctionNode):
    """Generate a grounded FNOL answer from the reranked policy passages.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        reranked_docs:    str  — JSON-serialised reranked passages
        validated_query:  str  — JSON-serialised normalised query
        runtime_settings: str  — JSON-serialised declared settings; the
                                 declared system_prompt_template is read here

    Output state keys (partial dict):
        grounded_answer: str  — JSON-serialised {claim_id, answer, citations,
                                grounded, intake_channel}
        status:          str
        error_log:       list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        reranked: list[dict[str, Any]] = from_json(state.get("reranked_docs"), [])
        query_obj: dict[str, Any] = from_json(state.get("validated_query"), {})
        claim_id = str(query_obj.get("claim_id", "unknown"))
        intake_channel = str(query_obj.get("intake_channel", "unknown"))

        settings: dict[str, Any] = from_json(state.get("runtime_settings"), {}) or {}
        prompt_template, template_resolved = _resolve_prompt_template(settings.get("system_prompt_template"))
        if not template_resolved:
            logger.warning(
                "GenerateAnswerNode: declared grounding-contract template not found: %s",
                prompt_template,
            )

        grounded = bool(reranked)
        if grounded:
            answer = _synthesise_grounded_answer(reranked)
            citations = [
                {
                    "doc_id": d.get("doc_id"),
                    "source": d.get("source"),
                    "policy_section": d.get("policy_section"),
                }
                for d in reranked
            ]
        else:
            answer = _UNGROUNDED_ANSWER
            citations = []

        grounded_answer: dict[str, Any] = {
            "claim_id": claim_id,
            "answer": answer,
            "citations": citations,
            "grounded": grounded,
            "intake_channel": intake_channel,
        }

        logger.info(
            "GenerateAnswerNode: claim_id=%s grounded=%s citations=%d",
            claim_id,
            grounded,
            len(citations),
        )
        emit_trace_event(
            "generate_answer_complete",
            {
                "claim_id": claim_id,
                "grounded": grounded,
                "citation_count": len(citations),
                "prompt_template": prompt_template,
                "prompt_template_resolved": template_resolved,
            },
            state,
        )

        return {
            "grounded_answer": to_json(grounded_answer),
            "status": AgentStatus.SUCCESS.value,
        }
