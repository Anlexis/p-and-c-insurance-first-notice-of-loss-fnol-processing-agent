# Template Design Specification — INS-C2-015 P&C Insurance FNOL Processing Agent

## Position in the AgentCore Architecture

| Item | Value |
|------|-------|
| Agent Class | FNOLProcessingAgent |
| L1 Base (framework base class) | AgentBaseGraph — direct framework inheritance |
| Pattern | Cat 2 — retrieval-augmented generation (two-layer nested workflow) |

- **Three-Layer Separation**:
  - State: flat TypedDict composition (no Pydantic — msgpack incompatible); dict/list fields are
    JSON-serialised strings
  - Node: framework inheritance (Template Method: override `execute(self, state) -> dict` only)
  - Graph: composition (`register_nodes()` for node substitution; Cat 2 nesting via `GraphNode`)

## Domain Context

First Notice of Loss (FNOL) intake and coverage-guidance agent for property-and-casualty
insurers. A claimant's FNOL query (claim id, claim type, question, policy number) is answered by
retrieving the relevant policy/coverage provisions from a policy knowledge base and generating a
**grounded** acknowledgment — the answer is derived strictly from the retrieved provisions, never
fabricated. Intake produces an acknowledgment only; final coverage determination and reserve
setting are completed by a licensed adjuster.

## Architecture Overview

### Backbone (outer AgentBaseGraph — fixed 5-node pipeline)

```
START → initialize → pre_process → main(GraphNode) → post_process → finalize → END
                                         ↓ (retry, max 3)
                                       pre_process
```

### Inner Domain Workflow (DomainWorkflowGraph — linear 5-node RAG pipeline)

```
START → input_validate → retrieve → rerank_filter → generate_answer
          → output_format → END
```

### Node Configuration

| Node | Class | File | Trust | Responsibility | Input Keys | Output Keys |
|------|-------|------|-------|---------------|------------|-------------|
| initialize | InitializeNode | framework | — | session init | — | session_id, schema_version |
| pre_process | PreProcessNode | src/nodes/pre_process_node.py | VERIFIED_EXTERNAL | trust gate + the caller-data contract | user_input, input_context | validated_input, enriched_context |
| main | FNOLRetrievalGraphNode | src/graph/graph.py | — | delegates to DomainWorkflowGraph | validated_input, input_context | fnol_response, grounded_answer, reranked_docs |
| post_process | PostProcessNode | src/nodes/post_process_node.py | ANONYMOUS | output boundary + set formatted_output | fnol_response | formatted_output, result |
| finalize | FinalizeNode | framework | — | response metadata | — | response_metadata, total_time_ms |
| input_validate (inner) | InputValidateNode | src/nodes/input_validate_node.py | ANONYMOUS | domain validation + claim_type normalise | validated_input, input_context | validated_query |
| retrieve (inner) | RetrieveNode | src/nodes/retrieve_node.py | ANONYMOUS | top_k policy-passage retrieval | validated_query, runtime_settings | retrieved_docs |
| rerank_filter (inner) | RerankFilterNode | src/nodes/rerank_filter_node.py | ANONYMOUS | rerank + score_threshold filter | retrieved_docs, runtime_settings | reranked_docs |
| generate_answer (inner) | GenerateAnswerNode | src/nodes/generate_answer_node.py | ANONYMOUS | grounded answer assembly | reranked_docs, validated_query, runtime_settings | grounded_answer |
| output_format (inner) | OutputFormatNode | src/nodes/output_format_node.py | ANONYMOUS | assemble final FNOL response document | grounded_answer | fnol_response, result |

`src/nodes/main_node.py` is not part of this pipeline: it is the concrete main-slot node used by
the flat-composition sample under `src/examples/`.

### Data Flow

```
user_input (JSON FNOL claim-query payload) + input_context (structured parameters)
    │
    ▼ PreProcessNode (VERIFIED_EXTERNAL — trust gate + caller contract)
validated_input (normalised JSON string)
enriched_context (JSON string)
    │
    ▼ FNOLRetrievalGraphNode → DomainWorkflowGraph
    │   (input_context bridged across the boundary — see below)
    │   InputValidateNode  → validated_query (JSON string)
    │   RetrieveNode       → retrieved_docs (JSON list, top_k)
    │   RerankFilterNode   → reranked_docs (JSON list, score_threshold)
    │   GenerateAnswerNode → grounded_answer (JSON {answer, citations, grounded})
    │   OutputFormatNode   → fnol_response (str), result (str)
    ▼ merge_output
fnol_response, grounded_answer, reranked_docs → outer state
    │
    ▼ PostProcessNode (ANONYMOUS — output boundary)
formatted_output (released document), result
```

### The caller-data contract

`POST /invoke` accepts two caller channels, and the contract for each is closed:

| Channel | Field | Rule |
|---------|-------|------|
| `input` | `claim_id` | required; `[A-Za-z0-9_-]{1,64}` (an inert identifier — it is printed into the acknowledgment) |
| `input` | `query` | required; non-empty string, ≤ 2000 characters |
| `input` | `claim_type` | optional; inert identifier, normalised to `auto` / `fire` / `accident` / `liability` / `unknown` |
| `input` | `policy_number` | optional; inert identifier |
| `input_context` | `channel` | optional; `[a-z0-9_]{1,32}` |

Rules that hold across the whole contract:

- **Unknown fields are refused, not ignored** on the `input` payload, and **dropped at the adapter**
  on `input_context`. Ignoring a key is not the same as removing it: an undeclared `input_context`
  key survives into the framework's first node result, where the output gate scans it.
- **Every declared payload field is a string.** The contract admits no numeric field at all, which
  is what makes "there is no caller-controlled number in this template" a property rather than a
  claim — and it is also what refuses the bare `NaN` / `Infinity` literals a JSON parser accepts.
- **Every number the agent acts on is a declared setting** (`top_k`, `score_threshold`), and each
  one goes through `finite_in_range` — real, finite, in range, or the node keeps its own default.
  A NaN threshold compares False against every score, so an unchecked one would silently change
  what the agent grounds an answer in.
- **Caller text is screened for instruction-override content** by the node that owns the contract,
  not by anything in front of it: chat-template control tokens (`<|…|>`, `[INST]`, `<<SYS>>`) as a
  class, plus anchored directive phrases. The screen runs on the raw body AND on the parsed payload
  depth-first, keys included, so an escaped directive that only exists after decoding is still seen.
- **Rejected values are never echoed.** Errors name the field; a field name is itself caller data
  and is echoed only when it is already inert.

### input_context across the nested boundary

The framework's `GraphNode.execute()` invokes the inner graph as
`subgraph.invoke(user_input, session_id=…, ctx=…)` and does not forward `input_context`, so an
inner-node read of `state["input_context"]` would always see `{}`. `src/graph/context_bridge.py`
bridges it through the two sanctioned hooks — `extract_input()` stashes it in a ContextVar before
the inner invoke, and `DomainWorkflowGraph._extra_initial_state()` seeds it into the inner initial
state. The ContextVar keeps the hand-off per-task, so concurrent invocations cannot see each
other's context.

### Runtime settings across the same boundary

The node contract is `execute(self, state) -> dict` and node constructors take no arguments, so a
declared setting cannot arrive as a constructor argument or as a per-invocation config argument.
`config/config.yaml` → `FNOLRetrievalGraphNode._parent_config()` (which validates every value) →
`DomainWorkflowGraph._extra_initial_state()` → `state["runtime_settings"]` → the RAG nodes. This is
the only live route; a node reading a `config` argument would read nothing.

### Input Payload Schema (user_input JSON)

```json
{
  "claim_id": "INS-FNOL-20260712-001",
  "claim_type": "auto",
  "query": "Is windshield damage from a road-debris strike covered under my comprehensive auto policy?",
  "policy_number": "PC-AUTO-88231"
}
```

### State Definition

| Field | Type | Purpose | Producer |
|-------|------|---------|----------|
| validated_input | NotRequired[Optional[str]] | Normalised FNOL query JSON string | PreProcessNode |
| enriched_context | NotRequired[Optional[str]] | JSON: {source, channel, claim_id} | PreProcessNode |
| runtime_settings | NotRequired[Optional[str]] | JSON: validated declared settings | DomainWorkflowGraph |
| validated_query | NotRequired[Optional[str]] | JSON: {claim_id, claim_type, query, policy_number, intake_channel} | InputValidateNode |
| retrieved_docs | NotRequired[Optional[str]] | JSON list of top_k scored passages | RetrieveNode |
| reranked_docs | NotRequired[Optional[str]] | JSON list of reranked+filtered passages | RerankFilterNode |
| grounded_answer | NotRequired[Optional[str]] | JSON: {claim_id, answer, citations, grounded, intake_channel} | GenerateAnswerNode |
| fnol_response | NotRequired[Optional[str]] | Final formatted FNOL response text | OutputFormatNode |
| result | NotRequired[Optional[str]] | Same as fnol_response (backbone convention) | OutputFormatNode / PostProcessNode |

**Serialization constraint**: all dict/list-valued fields use JSON-serialised `Optional[str]`.
`to_json()` / `from_json()` helpers are defined in `src/schemas/state.py` and used at every
producer/consumer boundary — one contract end-to-end.

**Prohibited**: re-declaring `formatted_output` (inherited from AgentState), credentials in State,
Pydantic models.

## Security Configuration

| Layer | Gate | Implementation |
|-------|------|---------------|
| Trust enforcement | caller trust | PreProcessNode `required_trust_level = VERIFIED_EXTERNAL` (inner nodes ANONYMOUS); the standalone entry point elevates a Bearer-authenticated caller |
| Input validation | caller contract | PreProcessNode (structural + inert identifiers + override screen) and InputValidateNode (domain rules); the inner node repeats the screen because it is separately reachable |
| Output boundary | credential screen | PostProcessNode module-level `_security_gate_output()` — the framework's own `detect_credentials()` plus domain assignment patterns, walking nested structures |
| Audit logging | trace events | `emit_trace_event()` in every node's `execute()` (at least one domain-specific event) |
| Credential handling | secrets | No credentials in State; the agent requires no secrets (`requires.secrets: []`) |

### The output invariant

The acknowledgment renders exactly three kinds of value: template text, text from the policy
knowledge base, and caller identifiers already restricted to the inert alphabet. The claimant's
question is **not** echoed into the document, and the pipeline produces **no monetary figure
anywhere** — there is no amount, limit, deductible value or reserve in any rendered output — so
there is no numeric grid to enforce and no rounding rule to state. The invariant that IS enforced
is the one this template can violate: no credential material leaves the boundary.

Two layers hold it, and they are not interchangeable:

1. the framework scans every node result and raises on a credential shape it recognises, which
   stops such material at the first node whose result carries it;
2. `PostProcessNode` screens the assembled document with **a superset of that same detector** —
   the framework's `detect_credentials()` plus domain assignment patterns (`password = …`), which
   the framework does not model. The superset matters for a specific reason: if the framework
   knows a pattern the node does not, the framework raises *inside* post_process, the node wrapper
   turns that into a bare error partial, and the node's own containment is discarded.

On a violation the node returns ERROR **and clears every output-bearing state field**
(`fnol_response`, `grounded_answer`, `reranked_docs`), because `AgentBaseGraph.get_output()` falls
back to `state["result"]` regardless of status — marking the status alone would still ship the
ungated document inside the error envelope. The caller receives a fixed withheld notice that does
not name which pattern matched.

**config/config.yaml** (runtime parameters; `config/agent.yaml` is the flat registration manifest
and carries no runtime block):
```yaml
max_retry: 3
timeout_s: 30
llm:
  system_prompt_template: prompts/fnol_coverage.j2
rag:
  top_k: 5
  score_threshold: 0.15
security:
  s3_gate_enabled: true
```

## RAG Tuning Notes

- `top_k` (RetrieveNode) — number of candidate policy passages pulled per query.
- `score_threshold` (RerankFilterNode) — minimum relevance score to keep a passage;
  below-threshold candidates are dropped so ungrounded answers are not fabricated. A passage whose
  score is missing or non-finite is dropped rather than defaulted.
- When no passage clears the threshold, GenerateAnswerNode returns an **ungrounded**
  acknowledgment routed to a licensed adjuster (never a coverage denial).

## Framework Utilization

### Shared Components Used
- [x] `InvocationContext` — session identity and caller trust
- [x] `detect_credentials()` — the framework's own detector, called by the output boundary
- [x] `emit_trace_event()` — at least one domain-specific event per node `execute()`
- [x] `to_json()` / `from_json()` helpers in `src/schemas/state.py`

### Composition Pattern

- **Pattern**: Cat 2 nested two-layer — GraphNode wrapping an inner BaseGraph
- **Outer graph**: `FNOLProcessingAgent(AgentBaseGraph)` — fixed 5-node backbone (`Graph` alias)
- **Inner graph**: `DomainWorkflowGraph(BaseGraph)` — 5-node linear RAG pipeline
- **Error propagation**: propagate (SubgraphError on inner failure; the outer backbone routes a
  failed run to finalize, and nothing is published)

## Import Isolation Confirmation
- [x] Template does not import the platform SDK
- [x] Import targets: `framework/` and `shared/` only

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| Framework base | AgentBaseGraph | AutonomousBaseGraph | AgentBaseGraph | Fixed sequential RAG pipeline; no autonomous reasoning loop |
| Composition pattern | flat | nested GraphNode | nested | 5 sequential RAG steps behind one main slot |
| Retrieval | inline in generation | separate Retrieve + Rerank nodes | separate nodes | Single responsibility; tunable top_k / score_threshold |
| State dict fields | bare dict | JSON-serialised str | JSON-serialised str | msgpack checkpoint safety |
| Answer generation | model call | deterministic grounded assembly | deterministic | The grounding contract is enforceable and testable without a model; `generation_mode: deterministic` in the manifest says so, and the declared prompt template states the same contract for a model-backed deployment |
| Settings delivery | node constructor arguments | seeded into inner state | inner state | Node constructors take no arguments and `execute(state)` takes no config argument |
| Ungrounded question | fabricate best guess | route to adjuster | route to adjuster | No coverage denial or fabrication at intake |
| Output invariant | numeric precision grid | credential containment | credential containment | Nothing monetary is rendered; the invariant enforced is the one this pipeline can violate |
