"""AgentCore Platform v1.0"""

# INS-C2-015 — RerankFilterNode
# Inner domain node 3 (RAG Rerank/Filter): re-order retrieved passages by
# relevance and drop those below the configured score_threshold.
#
# Deterministic rerank on the retrieval score with a lexical-density
# tie-breaker. Swapping in a cross-encoder reranker replaces _rerank_key only:
# the score_threshold gate and the passage contract are unchanged.
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

from src.schemas.state import from_json, to_json
from src.services.service import finite_in_range

logger = logging.getLogger(__name__)

_DEFAULT_SCORE_THRESHOLD = 0.15
_SCORE_THRESHOLD_BOUNDS = (0.0, 1.0)


def _rerank_key(doc: dict[str, Any]) -> tuple[float, int]:
    """Sort key: primary by score desc, tie-break by longer (denser) passage."""
    score = finite_in_range(doc.get("score"), 0.0, 1.0)
    length = len(str(doc.get("text", "")))
    return (score if score is not None else 0.0, length)


def _survives(doc: Any, score_threshold: float) -> bool:
    """True when *doc* carries a real, finite, in-range score at or above the bar.

    A passage whose score is missing, non-numeric or non-finite is DROPPED
    rather than defaulted: a NaN score compares False against any threshold, so
    defaulting it would let an unscored document ground an answer whenever the
    threshold happened to be 0.
    """
    if not isinstance(doc, dict):
        return False
    score = finite_in_range(doc.get("score"), 0.0, 1.0)
    return score is not None and score >= score_threshold


class RerankFilterNode(FunctionNode):
    """Rerank retrieved passages and filter below score_threshold.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        retrieved_docs:   str  — JSON-serialised list of scored passages
        runtime_settings: str  — JSON-serialised declared settings; the
                                 declared score_threshold is read from here

    Output state keys (partial dict):
        reranked_docs: str  — JSON-serialised filtered+sorted passages
        status:        str
        error_log:     list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        retrieved: list[dict[str, Any]] = from_json(state.get("retrieved_docs"), [])

        settings: dict[str, Any] = from_json(state.get("runtime_settings"), {}) or {}
        declared = finite_in_range(settings.get("score_threshold"), *_SCORE_THRESHOLD_BOUNDS)
        score_threshold = declared if declared is not None else _DEFAULT_SCORE_THRESHOLD

        if not isinstance(retrieved, list) or not retrieved:
            # Not an error: an empty candidate set yields an empty (but valid)
            # reranked set; GenerateAnswerNode handles the ungrounded case.
            logger.info("RerankFilterNode: no retrieved passages to rerank")
            emit_trace_event(
                "rerank_filter_complete",
                {"input_count": 0, "kept_count": 0, "score_threshold": score_threshold},
                state,
            )
            return {
                "reranked_docs": to_json([]),
                "status": AgentStatus.SUCCESS.value,
            }

        kept = [d for d in retrieved if _survives(d, score_threshold)]
        kept.sort(key=_rerank_key, reverse=True)

        logger.info(
            "RerankFilterNode: input=%d kept=%d score_threshold=%.3f",
            len(retrieved),
            len(kept),
            score_threshold,
        )
        emit_trace_event(
            "rerank_filter_complete",
            {
                "input_count": len(retrieved),
                "kept_count": len(kept),
                "score_threshold": score_threshold,
            },
            state,
        )

        return {
            "reranked_docs": to_json(kept),
            "status": AgentStatus.SUCCESS.value,
        }
