"""AgentCore Platform v1.0"""

# State must be a flat TypedDict — never a Pydantic BaseModel. LangGraph
# checkpoints use msgpack serialization; Pydantic objects cause silent
# corruption.  Extend AgentState with agent-specific fields only.  Do NOT add
# credentials, secrets, or Pydantic models.
#
# INS-C2-015 — P&C Insurance FNOL Processing Agent (Cat 2 nested RAG)
# Two-layer nested Cat 2 graph: outer backbone (AgentBaseGraph) + inner
# domain workflow (BaseGraph, RAG pipeline).  Fields below cover both layers.
#
# Serialization contract: all dict/list-valued fields are stored as JSON-
# serialized Optional[str].  Use to_json() / from_json() below at every
# producer and consumer node — one contract end-to-end.  Never type a
# dict/list field as a bare dict/list; that causes msgpack serialization
# failures at the checkpoint boundary.

import json
from typing import Any, NotRequired, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a value to a JSON string for State storage."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON string from State storage."""
    if value is None:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for INS-C2-015 FNOL Processing Agent.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.

    dict/list fields use JSON-serialized Optional[str].
    formatted_output is NOT re-declared here — it is inherited from AgentState.
    """

    # ------------------------------------------------------------------
    # Outer layer — set by PreProcessNode (pre_process backbone)
    # ------------------------------------------------------------------

    # Validated and normalised JSON string of the FNOL claim query payload.
    # Produced by PreProcessNode; consumed by inner InputValidateNode.
    validated_input: NotRequired[Optional[str]]

    # JSON-serialised channel/request metadata dict (stored as str).
    # Shape: {"source": str, "channel": str, "claim_id": str}
    enriched_context: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Inner layer — domain RAG nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # JSON-serialised mapping of the declared retrieval settings that survived
    # validation (top_k, score_threshold, system_prompt_template). Seeded by
    # DomainWorkflowGraph._extra_initial_state(); read by the RAG nodes. This
    # is the route by which a value declared in config/config.yaml reaches a
    # domain node — node constructors take no arguments and execute(state)
    # takes no config argument.
    runtime_settings: NotRequired[Optional[str]]

    # JSON-serialised normalised query object (stored as str).
    # Shape: {claim_id, claim_type, query, policy_number, intake_channel}
    validated_query: NotRequired[Optional[str]]

    # JSON-serialised list of retrieved KB passages (stored as str).
    # Shape: [{doc_id, text, score, source, policy_section}, ...]
    retrieved_docs: NotRequired[Optional[str]]

    # JSON-serialised list of reranked+filtered passages (stored as str).
    # Same shape as retrieved_docs, sorted desc by score, below-threshold dropped.
    reranked_docs: NotRequired[Optional[str]]

    # JSON-serialised grounded-answer object (stored as str).
    # Shape: {claim_id, answer, citations (list), grounded (bool)}
    grounded_answer: NotRequired[Optional[str]]

    # Final formatted FNOL response document (plain text, customer-ready).
    # Assembled by inner OutputFormatNode from grounded_answer.
    fnol_response: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Outer layer — set by PostProcessNode (post_process backbone)
    # ------------------------------------------------------------------

    # Primary result surfaced to the caller.
    # Set to the same content as fnol_response once the output boundary passes.
    # formatted_output (from AgentState) is also set by PostProcessNode.
    result: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Tracing / audit — framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: NotRequired[Optional[str]]
    correlation_id: NotRequired[Optional[str]]
    # node_history inherited from AgentState
