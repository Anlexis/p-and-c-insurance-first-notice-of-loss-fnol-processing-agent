# Test Specification — INS-C2-015 P&C Insurance FNOL Processing Agent

## 1. Test Strategy

- **Agent:** INS-C2-015 — P&C Insurance FNOL (First Notice of Loss) Processing
  Agent (Cat 2, retrieval-augmented, two-layer nested graph: outer
  `AgentBaseGraph` backbone + inner `DomainWorkflowGraph` `BaseGraph`).
- **Coverage target:** ≥ 90% of `src/nodes/`, `src/graph/` and `src/services/` branches.
- **Test types:** Unit (per node, per contract, graph wiring) · Proof-of-Boundary
  (framework security/serialization contracts, and end-to-end behaviour through
  the real ASGI entry point).
- **Framework provisioning:** `framework` (agenticstar-agentcore) is supplied by
  CI from the package registry. Tests import the real modules; there are no stub
  nodes.
- **Audit events:** `emit_trace_event` is patched at the node module level in unit
  tests to avoid audit-backend calls, never via a `sys.modules` stub (which would
  break the real `shared` package the framework loads at import time).

### RAG quality focus — grounding & abstention

INS-C2-015 is a retrieval-augmented FNOL assistant. The two properties that
matter most for correctness are asserted directly:

- **Grounding:** when the caller's question lexically matches the bundled P&C
  policy knowledge base, `RetrieveNode` → `RerankFilterNode` surface the relevant
  passages and `GenerateAnswerNode` produces a **grounded** answer whose citations
  come only from those passages (`grounded = True`, non-empty `citations`).
- **Abstention:** when no passage clears the `score_threshold` (an off-topic or
  unsupported question), the pipeline **abstains** — it never fabricates coverage.
  `GenerateAnswerNode` emits the fixed ungrounded acknowledgment ("routed to a
  licensed claims adjuster"), `grounded = False`, `citations = []`.

### Test file map

| File | Scope |
|------|-------|
| `tests/unit/test_nodes.py` | Every node + outer & inner graph wiring (grounding, abstention, declared-settings delivery, output boundary) |
| `tests/unit/test_caller_contract.py` | The caller-data contract: override screen (both directions), payload and context field rules, error-message hygiene, finite/bounded declared numbers, the rendered-document invariant |
| `tests/unit/test_main_node.py` | The flat-composition reference node + the trust gate (deny and admit) |
| `tests/unit/test_framework_compliance_tc06_tc07.py` | TC-06/TC-07: the framework's input/output gates cannot be overridden |
| `tests/proof_of_boundary/test_invoke_e2e.py` | End-to-end through the real ASGI `POST /invoke`: auth boundary, real grounded and abstained outcomes, the input_context bridge, validation rejection, credential-shaped context refusal, output-boundary containment |
| `tests/proof_of_boundary/test_pb_invoke_order.py` | PB-6 per-node + backbone invoke order (VERIFIED_EXTERNAL) + trust gate + payload alignment |
| `tests/proof_of_boundary/test_import_isolation.py` | PB-4 platform-SDK import isolation (AST scan) |
| `tests/proof_of_boundary/test_state_safety.py` | PB-2/PB-5 State msgpack/credential safety (AST scan) |
| `tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py` | PB-7 HITL interrupt-propagation (skipped — this template does not enable cross-boundary propagation) |

### Canonical valid payload (PB-6 `_VALID_PAYLOAD`)

The grounded auto-glass FNOL claim query used by the backbone invoke test and by
`deploy/invoke_payload.json` (the two MUST stay identical — asserted by
`test_invoke_payload_matches_pb6`):

```json
{
  "claim_id": "INS-FNOL-20260712-001",
  "claim_type": "auto",
  "policy_number": "PC-AUTO-88231",
  "query": "My windshield was cracked by road debris — is the glass damage covered under my comprehensive auto policy, and does the deductible apply?"
}
```

Grounding: the query overlaps the comprehensive / full-glass auto passages
(`AUTO-COMP-001`, `AUTO-GLASS-003`) above the `score_threshold` (0.15) ⇒
`grounded = True`, citations present.

## 2. Framework Compliance Tests (Mandatory)

| TC-ID | Test | Expected Result | Where |
|-------|------|----------------|-------|
| TC-01 | State contract: flat `TypedDict`, domain fields `NotRequired`, no Pydantic/dataclass | AST scan: 0 violations | `test_state_safety.py` |
| TC-02 | Invalid/empty/non-JSON input rejected at PreProcessNode | `status=error`, error_log populated | `TestPreProcessNode`, `TestPayloadContract` |
| TC-03 | No credential material in State | CI credential scan: 0 violations | CI + `test_state_safety.py` |
| TC-04 | `execute(self, state)` contract — no private invoke hook | Signature `(self, state)` | `test_execute_signature_is_state_first`, `test_main_node.py` |
| TC-05 | `emit_trace_event()` called inside each node `execute()` | ≥1 domain event per node | `scripts/check_audit_trace.py` (CI gate) |
| TC-06 | The framework input gate cannot be overridden | TypeError at class definition | `test_framework_compliance_tc06_tc07.py` |
| TC-07 | The framework output gate cannot be overridden | TypeError at class definition | `test_framework_compliance_tc06_tc07.py` |
| TC-08 | `required_trust_level` enforced in `__call__` before `execute()` | ANONYMOUS caller → refused; VERIFIED_EXTERNAL → admitted | `TestTrustGate` |
| TC-08a | Outer `PreProcessNode` = VERIFIED_EXTERNAL; inner nodes + post_process = ANONYMOUS | trust levels asserted per node | `test_trust_level_*` |
| TC-11 | Output boundary on post_process | credential pattern → withheld + `status=error` + every output-bearing field cleared; clean → released | `TestPostProcessNode`, `TestOutputBoundaryContainment` |

## 3. Proof-of-Boundary Tests (Mandatory)

| PB-ID | Boundary | Test | Expected Result | Where |
|-------|----------|------|----------------|-------|
| PB-2 | State serialization | AST scan of `src/schemas/state.py` | primitives only; no Pydantic/dataclass | `test_state_safety.py` |
| PB-4 | Import isolation | AST scan of `src/` | 0 platform-SDK imports | `test_import_isolation.py` |
| PB-5 | Checkpoint safety | no credential-named fields / prohibited types in State | inspection pass | `test_state_safety.py` |
| PB-6 | Invoke execution order (per node) | `__call__`: node_start → trust gate → input gate → `execute()` → output gate → node_complete | order verified for every `src/nodes/` class | `TestInvokeOrder` |
| PB-6b | Backbone invoke order | full `Graph().invoke(_VALID_PAYLOAD, ctx=VERIFIED_EXTERNAL)` | `status=success`; node_history = `[Initialize, PreProcess, FNOLRetrievalGraphNode, PostProcess, Finalize]` | `TestBackboneInvokeOrder`, `TestInvokeEndToEnd` |
| PB-6c | Real external caller | `InvocationContext(caller_trust_level=VERIFIED_EXTERNAL)` — **never** `for_internal()` | inner ANONYMOUS nodes accept the passthrough trust; SUCCESS end-to-end | `TestBackboneInvokeOrder` |
| PB-6d | Payload alignment | `deploy/invoke_payload.json["input"] == _VALID_PAYLOAD` | the deployment smoke invoke exercises the PB-6 payload | `test_invoke_payload_matches_pb6` |
| PB-7 | HITL interrupt propagation | `FNOLRetrievalGraphNode.propagate_hitl=False`, no cross-boundary interrupt() checkpoint | skipped with reason | `test_pb7_hitl_interrupt_propagation.py` |
| PB-8 | Entry-point auth | `POST /invoke` without / with a wrong / with a non-ASCII Bearer token | 401, nothing published | `TestEntryPointAuth` |
| PB-9 | input_context reaches the inner graph | a caller `channel` value rendered by the inner pipeline | the document changes with the caller value | `TestInputContextReachesTheInnerGraph` |
| PB-10 | Credential-shaped context refusal | a `Bearer …` / JWT / `sk-…` value on `input_context.channel` | 400 naming the field, never the value | `TestCredentialShapedContextIsRefusedReadably` |

## 4. Business Logic Tests

| BL-ID | Test | Input | Expected Result | Where |
|-------|------|-------|----------------|-------|
| BL-01 | Happy-path grounded FNOL response | `_VALID_PAYLOAD` (auto-glass claim) | `FIRST NOTICE OF LOSS` + claim_id present; grounded response with citations | `TestInvokeEndToEnd`, `test_inner_graph_invoke_grounds_answer` |
| BL-02 | RAG retrieval relevance | validated auto-glass query | comprehensive/glass passages retrieved, scores sorted desc; `AUTO-COMP-001` surfaced | `TestRetrieveNode` |
| BL-03 | Rerank threshold filter | scored passages `{0.90, 0.20, 0.05}`, threshold 0.15 | passages below threshold dropped, kept sorted desc | `TestRerankFilterNode` |
| BL-04 | Grounded generation | reranked passages present | `grounded=True`, citations mirror passages, no fabrication | `test_grounded_answer_cites_passages` |
| BL-05 | **Abstention** (no supporting passage) | off-topic query / empty reranked set | `grounded=False`, `citations=[]`, "routed to adjuster" acknowledgment | `test_abstains_when_no_passages`, `test_an_unsupported_question_abstains_instead_of_fabricating` |
| BL-06 | Claim-type normalisation | `car`→auto, `home`→fire, `injury`→accident, unknown→`unknown` | canonical claim_type in validated_query | `test_claim_type_aliases_are_normalised` |
| BL-07 | Response assembly + citations | grounded answer w/ citations | header, claim id, intake channel, `policy-grounded`, CITATIONS block | `TestOutputFormatNode` |
| BL-08 | Knowledge-base immutability | any retrieval | module-global `_POLICY_KB` unchanged after execute() | `test_does_not_mutate_module_kb` |
| BL-09 | Graph key coupling | inner `get_output` ↔ outer `merge_output` | 4 coupled keys mapped; `merge_output` returns changed keys only | `TestOuterGraphComposition`, `TestInnerDomainGraph` |
| BL-10 | Declared settings are live | `config/config.yaml` `top_k` / `score_threshold` | the declared value reaches the node and changes the result | `test_declared_top_k_limits_results`, `test_the_manifest_declaration_survives_the_forwarding_step` |

### Negative / boundary cases

| Case | Where enforced | Expected |
|------|----------------|----------|
| empty `user_input` | PreProcessNode | `status=error`, "empty" |
| payload over 32 KB | PreProcessNode | `status=error`, "limit" |
| invalid JSON | PreProcessNode | `status=error`, "invalid JSON" |
| JSON root not an object | PreProcessNode | `status=error`, "object" |
| unknown payload field | PreProcessNode | `status=error`, "unsupported payload fields" |
| non-string payload field (incl. raw `NaN`) | PreProcessNode | `status=error` |
| missing / empty `query` | PreProcessNode, InputValidateNode | `status=error`, "query" |
| `query` over 2000 chars | PreProcessNode | `status=error`, "2000" |
| `claim_id` / `policy_number` outside `[A-Za-z0-9_-]{1,64}` | PreProcessNode, InputValidateNode | `status=error`, field named |
| `input_context.channel` outside `[a-z0-9_]{1,32}` | PreProcessNode | `status=error`, "channel" |
| unknown `input_context` key | adapter (dropped) / PreProcessNode (refused) | not forwarded to the graph |
| credential-shaped `input_context` value | `src/api/server.py` | HTTP 400 naming the field |
| chat-template control token (`<\|…\|>`, `[INST]`, `<<SYS>>`) | PreProcessNode, InputValidateNode | `status=error` |
| instruction-override directive (raw or `\u`-escaped) | PreProcessNode, InputValidateNode | `status=error` |
| hostile field NAME | PreProcessNode | `status=error`, reported positionally |
| non-finite declared `top_k` / `score_threshold` | RetrieveNode, RerankFilterNode | falls back to the documented default |
| passage with a non-finite score | RerankFilterNode | dropped, never grounds an answer |
| missing `validated_query` | RetrieveNode | `status=error`, "validated_query" |
| empty retrieved set | RerankFilterNode | `status=success`, `reranked_docs=[]` |
| missing `grounded_answer` | OutputFormatNode | `status=error` |
| non-inert identifier at the render boundary | OutputFormatNode | substituted, never rendered |
| empty `fnol_response` | PostProcessNode | fallback message, `status=success` |
| credential in the assembled document | PostProcessNode | withheld notice, `status=error`, output-bearing fields cleared |

## 5. Test Execution Summary

- Execution: `pytest tests/` against the framework wheel CI installs.
- Total: 181 tests — **180 passed, 1 skipped** (PB-7, by design; this template does
  not enable cross-boundary HITL propagation).
- Static checks under `scripts/` (repository structure, import isolation, node
  composition, credential scan, trust levels, audit-trace coverage, manifest
  schema, dependency pinning, licence) — all PASS.
- Coverage: every node exercised on both the grounded and the abstained path, the
  contract exercised in both directions (hostile refused, legitimate accepted),
  and the output boundary exercised through the real entry point.
