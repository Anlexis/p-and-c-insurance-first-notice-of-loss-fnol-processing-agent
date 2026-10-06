"""AgentCore Platform v1.0"""

# INS-C2-015 — InputValidateNode
# Inner domain node 1 (first): domain-level validation of the FNOL claim query.
#
# Distinct from PreProcessNode (trust enforcement + the structural caller
# contract): this node applies domain business rules — claim_type
# normalisation, query length bounds, and the normalised query object the RAG
# pipeline consumes.
#
# Inner node — ANONYMOUS trust. The outer PreProcessNode (VERIFIED_EXTERNAL)
# already enforced trust; inner nodes must be ANONYMOUS so the outer
# InvocationContext passes through the GraphNode boundary without rejection.
#
# The caller-facing screens are NOT repeated here as a substitute for the ones
# in PreProcessNode — they are repeated because this node can also be reached
# by an inner-graph invoke, and a node that owns a contract enforces it itself.
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import json
import logging
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import to_json
from src.services.service import is_identifier, is_inert_token, screen_structure


# The Marketplace runner seeds input_context with its own conversation history on every
# invocation (shared/bootstrap/marketplace_app.py); the caller neither sends that key nor can
# suppress it, and the build_input_context hook can only overwrite its value, never remove it.
# It is platform plumbing rather than caller data, so it is dropped here, before the caller
# contract runs: the unknown-field guard below stays strict for everything a caller can
# actually send, and no value screen is ever asked to judge a transcript that contains this
# agent's own earlier answers. The value may also be None, which this tolerates.
_PLATFORM_CONTEXT_KEYS = frozenset({"conversation_history"})


def _without_platform_context(raw: Any) -> Any:
    """The caller-supplied half of input_context, platform-injected keys removed."""
    if not isinstance(raw, dict):
        return raw
    return {k: v for k, v in raw.items() if k not in _PLATFORM_CONTEXT_KEYS}


logger = logging.getLogger(__name__)

# Canonical P&C claim types.  Aliases are normalised to these labels so the
# retriever can bias toward the correct policy line.
_CLAIM_TYPE_ALIASES: dict[str, str] = {
    "auto": "auto",
    "car": "auto",
    "motor": "auto",
    "vehicle": "auto",
    "fire": "fire",
    "property": "fire",
    "home": "fire",
    "accident": "accident",
    "injury": "accident",
    "liability": "liability",
    "unknown": "unknown",
}

_MAX_QUERY_CHARS = 2000

# Intake channel rendered into the acknowledgment header. Restricted to the
# inert token alphabet; anything else degrades to the default rather than
# reaching the response document.
_DEFAULT_CHANNEL = "unknown"


def _normalise_claim_type(raw: Any) -> str:
    """Normalise a raw claim_type string to a canonical label."""
    key = str(raw or "unknown").strip().lower()
    return _CLAIM_TYPE_ALIASES.get(key, "unknown")


class InputValidateNode(FunctionNode):
    """Domain validation of the FNOL claim query for INS-C2-015.

    Applies business-rule checks beyond the structural contract in
    PreProcessNode: claim_type normalisation, query length bounds, and
    construction of the normalised query object consumed by RetrieveNode.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        validated_input: str  — normalised JSON string from PreProcessNode.
                                Falls back to user_input when absent.
        input_context:   dict — bridged structured invocation parameters

    Output state keys (partial dict):
        validated_query: str   — JSON-serialised normalised query object
        status:          str
        error_log:       list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        raw = state.get("validated_input") or state.get("user_input", "")

        # ── Parse ─────────────────────────────────────────────────────────────
        try:
            payload = json.loads(raw) if isinstance(raw, str) else {}
        except (json.JSONDecodeError, ValueError):
            logger.error("InputValidateNode: JSON parse error")
            return self._reject("json_parse_error", "invalid JSON payload", state)

        if not isinstance(payload, dict):
            return self._reject("payload_not_dict", "payload is not a JSON object", state)

        # ── Instruction-override screen (values AND keys, depth-first) ────────
        finding = screen_structure(payload)
        if finding:
            return self._reject(
                "instruction_override",
                "payload rejected: disallowed instruction content",
                state,
                detail=finding,
            )

        query = str(payload.get("query", "")).strip()
        if not query:
            return self._reject("empty_query", "query is missing or empty", state)

        if len(query) > _MAX_QUERY_CHARS:
            query = query[:_MAX_QUERY_CHARS]

        # ── Identifier alphabet ───────────────────────────────────────────────
        claim_id = payload.get("claim_id")
        if not is_identifier(claim_id):
            return self._reject(
                "invalid_claim_id",
                "claim_id must be 1-64 characters of letters, digits, '-' or '_'",
                state,
            )
        raw_policy_number = payload.get("policy_number")
        if raw_policy_number is not None and not is_identifier(raw_policy_number):
            return self._reject(
                "invalid_policy_number",
                "policy_number must be 1-64 characters of letters, digits, '-' or '_'",
                state,
            )

        claim_type = _normalise_claim_type(payload.get("claim_type"))
        policy_number = raw_policy_number or None

        raw_context = _without_platform_context(state.get("input_context")) or {}
        channel = raw_context.get("channel") if isinstance(raw_context, dict) else None
        intake_channel = channel if is_inert_token(channel) else _DEFAULT_CHANNEL

        validated_query: dict[str, Any] = {
            "claim_id": claim_id,
            "claim_type": claim_type,
            "query": query,
            "policy_number": policy_number,
            "intake_channel": intake_channel,
        }

        logger.info(
            "InputValidateNode: claim_id=%s claim_type=%s query_len=%d",
            claim_id,
            claim_type,
            len(query),
        )
        emit_trace_event(
            "input_validate_complete",
            {
                "claim_id": claim_id,
                "claim_type": claim_type,
                "query_length": len(query),
                "intake_channel": intake_channel,
            },
            state,
        )

        return {
            "validated_query": to_json(validated_query),
            "status": AgentStatus.SUCCESS.value,
        }

    @staticmethod
    def _reject(reason: str, message: str, state: AgentState, detail: Any = None) -> dict[str, Any]:
        """Fail closed with an audit event, naming the field and never the value."""
        event: dict[str, Any] = {"reason": reason}
        if detail is not None:
            event["detail"] = detail
        emit_trace_event("input_validate_failed", event, state)
        return {
            "status": AgentStatus.ERROR.value,
            "error_log": [f"InputValidateNode: {message}"],
            # The runner surfaces `formatted_output or result` as `output`. A reason left only in
            # error_log reaches no one: the terminal result carries just `status`, and get_output()
            # does not copy error_log out of the graph -- the caller sees a blank spinner.
            "formatted_output": "Request could not be completed. " + (f"InputValidateNode: {message}"),
        }
