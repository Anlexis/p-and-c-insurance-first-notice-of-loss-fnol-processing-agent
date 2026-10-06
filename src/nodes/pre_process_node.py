"""AgentCore Platform v1.0"""

# INS-C2-015 — PreProcessNode
# Outer backbone pre_process slot: trust enforcement + the caller-data contract.
#
# Responsibilities:
#   - Enforce VERIFIED_EXTERNAL trust (required_trust_level)
#   - Reject empty / non-JSON input early (fail-fast)
#   - Enforce the FNOL claim-query contract: required fields present, every
#     caller string bounded, identifiers restricted to an inert alphabet
#   - Refuse instruction-override content in the node that owns the caller
#     contract, so refusal does not depend on any gate in front of it
#   - Validate the structured input_context channel field by field
#   - Write validated_input (normalised JSON string) + enriched_context to State
#   - Emit an audit event for every validation decision
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import json
import logging
from typing import Any, ClassVar, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import to_json
from src.services.service import (
    is_identifier,
    is_inert_token,
    safe_field_name,
    screen_structure,
    screen_text,
)


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

# Required top-level keys for a valid FNOL claim-query payload.
# claim_id identifies the claim; query is the coverage/processing question
# the agent must answer, grounded in the policy knowledge base.
_REQUIRED_CLAIM_KEYS = frozenset({"claim_id", "query"})

# The complete caller contract. Every key the payload may carry is listed here
# with the check it must pass; anything else is refused rather than ignored,
# because an ignored key is still caller data travelling into the pipeline.
_ALLOWED_CLAIM_KEYS = frozenset({"claim_id", "claim_type", "query", "policy_number"})

# Structural caps. The question is the only free-text field the contract
# admits, and it is bounded here as well as at the domain node.
_MAX_QUERY_CHARS = 2000
_MAX_PAYLOAD_BYTES = 32_768

# The structured invocation channel. Values are inert lowercase tokens only:
# intake_channel is rendered into the acknowledgment, so free text there would
# be caller-controlled output. Unknown keys are refused, not silently ignored.
_ALLOWED_CONTEXT_KEYS = frozenset({"channel"})


class PreProcessNode(FunctionNode):
    """Caller-data contract for INS-C2-015.

    Validates the caller-supplied FNOL claim-query payload before the domain
    RAG workflow runs.  This is the outer backbone's pre_process slot — the
    only node with VERIFIED_EXTERNAL trust so that unauthenticated or
    anonymous callers are rejected here (fail-fast; inner domain nodes carry
    ANONYMOUS trust and never see untrusted input directly).

    Input state keys:
        user_input:    str  — caller-supplied JSON FNOL claim-query payload
        input_context: dict — structured invocation parameters (inert tokens)

    Output state keys (partial dict):
        validated_input:  str        — normalised JSON string (re-serialised)
        enriched_context: str        — JSON-serialised channel metadata
        status:           str        — AgentStatus.SUCCESS or ERROR
        error_log:        list[str]  — set only on ERROR
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState) -> dict[str, Any]:
        user_input = state.get("user_input", "")
        raw_context = _without_platform_context(state.get("input_context")) or {}

        # ── Emptiness check ───────────────────────────────────────────────────
        if not user_input or not isinstance(user_input, str) or not user_input.strip():
            logger.warning("PreProcessNode: user_input is empty or missing")
            return self._reject("empty_input", "user_input is empty or missing", state)

        if len(user_input.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
            return self._reject(
                "payload_too_large",
                f"payload exceeds the {_MAX_PAYLOAD_BYTES}-byte limit",
                state,
            )

        # ── Instruction-override screen, raw form ─────────────────────────────
        # Screened before parsing so a control token is caught even when the
        # body is not valid JSON, and again after parsing (below) so an escaped
        # directive that only exists once decoded is caught too.
        finding = screen_text(user_input)
        if finding:
            return self._reject(
                "instruction_override",
                "user_input rejected: disallowed instruction content",
                state,
                detail=finding,
            )

        # ── JSON parse ────────────────────────────────────────────────────────
        try:
            payload = json.loads(user_input.strip())
        except (json.JSONDecodeError, ValueError):
            # The parser message quotes the offending input, so it is not
            # carried into the error the caller sees.
            logger.warning("PreProcessNode: JSON parse failed")
            return self._reject("json_parse_error", "invalid JSON payload", state)

        if not isinstance(payload, dict):
            return self._reject("payload_not_object", "JSON root must be an object", state)

        # ── Instruction-override screen, parsed form (values AND keys) ────────
        finding = screen_structure(payload)
        if finding:
            return self._reject(
                "instruction_override",
                "user_input rejected: disallowed instruction content",
                state,
                detail=finding,
            )

        # ── Contract: no unknown fields ───────────────────────────────────────
        unknown = [
            safe_field_name(key, position)
            for position, key in enumerate(payload, start=1)
            if key not in _ALLOWED_CLAIM_KEYS
        ]
        if unknown:
            return self._reject(
                "unknown_fields",
                f"unsupported payload fields: {sorted(unknown)}",
                state,
            )

        # ── Required field check ──────────────────────────────────────────────
        missing = _REQUIRED_CLAIM_KEYS - payload.keys()
        if missing:
            return self._reject(
                "missing_required_fields",
                f"missing required fields: {sorted(missing)}",
                state,
                detail={"missing": sorted(missing)},
            )

        # ── Every declared field is a string ──────────────────────────────────
        # The contract admits no numeric field at all, so there is no
        # caller-controlled number anywhere in this template: refusing
        # non-strings here is what makes that statement true rather than
        # aspirational, and it also refuses the JSON literals NaN / Infinity,
        # which Python's json module parses happily.
        non_strings = [key for key in payload if not isinstance(payload[key], str)]
        if non_strings:
            return self._reject(
                "non_string_field",
                f"fields must be strings: {sorted(non_strings)}",
                state,
            )

        # ── Query bounds ──────────────────────────────────────────────────────
        query = payload["query"].strip()
        if not query:
            return self._reject("empty_query", "query is empty", state)
        if len(query) > _MAX_QUERY_CHARS:
            return self._reject(
                "query_too_long",
                f"query exceeds the {_MAX_QUERY_CHARS}-character limit",
                state,
            )

        # ── Identifier alphabet ───────────────────────────────────────────────
        # claim_id is printed into the acknowledgment; policy_number travels
        # with the normalised query. Both are locked to an inert alphabet so
        # neither can carry markup or instructions into the response document.
        claim_id = payload["claim_id"]
        if not is_identifier(claim_id):
            return self._reject(
                "invalid_claim_id",
                "claim_id must be 1-64 characters of letters, digits, '-' or '_'",
                state,
            )
        policy_number = payload.get("policy_number")
        if policy_number is not None and not is_identifier(policy_number):
            return self._reject(
                "invalid_policy_number",
                "policy_number must be 1-64 characters of letters, digits, '-' or '_'",
                state,
            )
        claim_type = payload.get("claim_type")
        if claim_type is not None and not is_identifier(claim_type):
            return self._reject(
                "invalid_claim_type",
                "claim_type must be 1-64 characters of letters, digits, '-' or '_'",
                state,
            )

        # ── Structured invocation channel ─────────────────────────────────────
        context_error = self._validate_context(raw_context)
        if context_error is not None:
            return self._reject("invalid_input_context", context_error, state)
        channel = str(raw_context.get("channel", "")) or "unknown"

        # ── Success ───────────────────────────────────────────────────────────
        normalised = {key: payload[key] for key in sorted(payload)}
        normalised["query"] = query
        normalised_json = json.dumps(normalised, ensure_ascii=False)

        logger.info(
            "PreProcessNode: validated claim_id=%s payload_keys=%d",
            claim_id,
            len(normalised),
        )
        emit_trace_event(
            "pre_process_validated",
            {
                "claim_id": claim_id,
                "payload_keys": sorted(normalised.keys()),
                "channel": channel,
            },
            state,
        )

        return {
            "validated_input": normalised_json,
            "enriched_context": to_json(
                {
                    "source": "FNOLProcessingAgent",
                    "channel": channel,
                    "claim_id": claim_id,
                }
            ),
            "status": AgentStatus.SUCCESS.value,
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_context(raw_context: Any) -> Optional[str]:
        """Validate the structured invocation channel; return an error or None."""
        if not isinstance(raw_context, dict):
            return "input_context must be an object"
        unknown = [
            safe_field_name(key, position)
            for position, key in enumerate(raw_context, start=1)
            if key not in _ALLOWED_CONTEXT_KEYS
        ]
        if unknown:
            return f"unsupported input_context fields: {sorted(unknown)}"
        if "channel" in raw_context and not is_inert_token(raw_context["channel"]):
            return "input_context.channel must be 1-32 characters of [a-z0-9_]"
        return None

    @staticmethod
    def _reject(
        reason: str,
        message: str,
        state: AgentState,
        detail: Any = None,
    ) -> dict[str, Any]:
        """Fail closed with an audit event.

        The message names the field, never the value: a rejected payload is
        exactly the payload that must not be echoed back into a log line or an
        error response.
        """
        event: dict[str, Any] = {"reason": reason}
        if detail is not None:
            event["detail"] = detail
        emit_trace_event("pre_process_validation_failed", event, state)
        return {
            "status": AgentStatus.ERROR.value,
            "error_log": [f"PreProcessNode: {message}"],
            # The runner surfaces `formatted_output or result` as `output`. A reason left only in
            # error_log reaches no one: the terminal result carries just `status`, and get_output()
            # does not copy error_log out of the graph -- the caller sees a blank spinner.
            "formatted_output": "Request could not be completed. " + (f"PreProcessNode: {message}"),
        }
