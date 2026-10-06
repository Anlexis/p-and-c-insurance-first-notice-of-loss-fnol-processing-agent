"""AgentCore Platform v1.0"""

# INS-C2-015 — RetrieveNode
# Inner domain node 2 (RAG Retrieve): retrieve the top_k most relevant
# policy/coverage KB passages for the validated FNOL query.
#
# This implementation scores with deterministic keyword overlap over a bundled
# P&C policy corpus. Swapping in a vector store (embeddings + ANN search)
# replaces _score_passage and _POLICY_KB only: the top_k budget, the passage
# shape and the state contract are unchanged.
#
# Inner node — ANONYMOUS trust (see DomainWorkflowGraph).
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json, to_json
from src.services.service import finite_in_range

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 5
_TOP_K_BOUNDS = (1.0, 50.0)

# Bundled P&C policy knowledge base.  Each passage is a coverage clause a
# claims-intake agent must be able to ground an FNOL answer in.  The passage
# shape ({doc_id, source, policy_section, claim_types, text}) is the stable
# contract a vector-store retrieval must also satisfy.
_POLICY_KB: list[dict[str, Any]] = [
    {
        "doc_id": "AUTO-COMP-001",
        "source": "Comprehensive Auto Policy",
        "policy_section": "Section III — Comprehensive Coverage",
        "claim_types": ["auto"],
        "text": (
            "Comprehensive coverage pays for damage to the insured vehicle from "
            "causes other than collision, including glass breakage, windshield and "
            "road-debris strikes, fire, theft, vandalism, hail, and falling objects, "
            "subject to the comprehensive deductible stated on the declarations page."
        ),
    },
    {
        "doc_id": "AUTO-COLL-002",
        "source": "Comprehensive Auto Policy",
        "policy_section": "Section II — Collision Coverage",
        "claim_types": ["auto"],
        "text": (
            "Collision coverage pays for damage to the insured vehicle caused by "
            "impact with another vehicle or object, or by overturn, regardless of "
            "fault, less the collision deductible."
        ),
    },
    {
        "doc_id": "AUTO-GLASS-003",
        "source": "Comprehensive Auto Policy",
        "policy_section": "Endorsement A — Full Glass",
        "claim_types": ["auto"],
        "text": (
            "The full-glass endorsement waives the deductible for windshield and "
            "window glass repair or replacement when comprehensive coverage applies. "
            "Repair is preferred over replacement where safe and feasible."
        ),
    },
    {
        "doc_id": "FIRE-DWELL-004",
        "source": "Homeowners / Fire Property Policy",
        "policy_section": "Coverage A — Dwelling",
        "claim_types": ["fire"],
        "text": (
            "Dwelling coverage pays for direct physical loss to the residence from "
            "fire, lightning, smoke, and windstorm, up to the Coverage A limit, less "
            "the applicable deductible. Losses from flood and earth movement are excluded."
        ),
    },
    {
        "doc_id": "ACC-BODILY-005",
        "source": "Personal Accident Policy",
        "policy_section": "Part 1 — Accidental Bodily Injury",
        "claim_types": ["accident"],
        "text": (
            "Accidental bodily injury benefits are payable for medical expenses and "
            "scheduled indemnity resulting from a sudden, unforeseen accident, provided "
            "the first notice of loss is filed within the reporting window."
        ),
    },
    {
        "doc_id": "GEN-FNOL-006",
        "source": "Claims Handling Manual",
        "policy_section": "FNOL Intake Procedure",
        "claim_types": ["auto", "fire", "accident", "liability", "unknown"],
        "text": (
            "First Notice of Loss (FNOL) must capture the claimant, policy number, "
            "date and description of loss, and supporting documents. The intake agent "
            "acknowledges receipt, assigns a claim number, and routes to coverage "
            "verification and reserve setting before adjuster assignment."
        ),
    },
    {
        "doc_id": "GEN-EXCL-007",
        "source": "Claims Handling Manual",
        "policy_section": "General Exclusions",
        "claim_types": ["auto", "fire", "accident", "liability", "unknown"],
        "text": (
            "No coverage applies to intentional loss, normal wear and tear, mechanical "
            "breakdown, or loss occurring outside the policy period. Coverage disputes "
            "are escalated to a licensed adjuster; the intake agent must not deny a claim."
        ),
    },
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "of",
        "to",
        "for",
        "and",
        "or",
        "in",
        "on",
        "under",
        "my",
        "our",
        "your",
        "does",
        "do",
        "it",
        "this",
        "that",
        "with",
        "from",
        "by",
        "be",
        "can",
        "will",
        "would",
        "i",
        "we",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stopwords removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _score_passage(query_tokens: list[str], passage: dict[str, Any], claim_type: str) -> float:
    """Keyword-overlap relevance score in [0, 1], with a claim-type match bonus."""
    if not query_tokens:
        return 0.0
    passage_tokens = set(_tokenize(str(passage["text"])))
    overlap = sum(1 for t in query_tokens if t in passage_tokens)
    base = overlap / len(query_tokens)
    # Claim-type match is a boost, NOT a free pass: it is kept below the default
    # score_threshold (0.15) so a passage still needs genuine lexical relevance
    # to survive the rerank filter (type-bonus alone must not ground an answer).
    bonus = 0.10 if claim_type in passage.get("claim_types", []) else 0.0
    return round(min(base + bonus, 1.0), 4)


class RetrieveNode(FunctionNode):
    """Retrieve top_k candidate policy passages for the FNOL query.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        validated_query:  str  — JSON-serialised normalised query
        runtime_settings: str  — JSON-serialised declared settings; the
                                 declared top_k is read from here (seeded by
                                 DomainWorkflowGraph._extra_initial_state)

    Output state keys (partial dict):
        retrieved_docs: str  — JSON-serialised list of scored passages
        status:         str
        error_log:      list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        query_obj: dict[str, Any] = from_json(state.get("validated_query"), {})
        query = str(query_obj.get("query", "")).strip()
        claim_type = str(query_obj.get("claim_type", "unknown"))

        settings: dict[str, Any] = from_json(state.get("runtime_settings"), {}) or {}
        declared_top_k = finite_in_range(settings.get("top_k"), *_TOP_K_BOUNDS)
        top_k = int(declared_top_k) if declared_top_k is not None else _DEFAULT_TOP_K

        if not query:
            logger.error("RetrieveNode: validated_query missing/empty in state")
            emit_trace_event("retrieve_failed", {"reason": "missing_query"}, state)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["RetrieveNode: validated_query missing or empty"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("RetrieveNode: validated_query missing or empty"),
            }

        query_tokens = _tokenize(query)

        # Build a fresh scored list — never mutate the module-global _POLICY_KB.
        scored: list[dict[str, Any]] = []
        for passage in _POLICY_KB:
            score = _score_passage(query_tokens, passage, claim_type)
            if score <= 0.0:
                continue
            scored.append(
                {
                    "doc_id": passage["doc_id"],
                    "text": passage["text"],
                    "score": score,
                    "source": passage["source"],
                    "policy_section": passage["policy_section"],
                }
            )

        scored.sort(key=lambda d: float(d["score"]), reverse=True)
        retrieved = scored[:top_k]

        logger.info(
            "RetrieveNode: claim_type=%s candidates=%d retrieved=%d top_k=%d",
            claim_type,
            len(scored),
            len(retrieved),
            top_k,
        )
        emit_trace_event(
            "retrieve_complete",
            {
                "claim_type": claim_type,
                "candidate_count": len(scored),
                "retrieved_count": len(retrieved),
                "top_k": top_k,
            },
            state,
        )

        return {
            "retrieved_docs": to_json(retrieved),
            "status": AgentStatus.SUCCESS.value,
        }
