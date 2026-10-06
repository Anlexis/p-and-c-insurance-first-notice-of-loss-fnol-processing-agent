# FNOL Claims Intake Processing Agent

AI agent for processing first notice of loss for property and casualty insurance claims, built with Agentic Star.

> **Category**: Cat 2 (domain-specific pipeline — retrieval-augmented)
> **Industry**: Insurance
> **Template ID**: INS-C2-015

## Overview

Answers a claimant's First Notice of Loss (FNOL) coverage question from the policy wording, and
refuses to guess when the wording does not cover it.

A claim query arrives as a small JSON object — claim id, the claimant's question, and optionally a
claim type and policy number. The agent validates it, retrieves the most relevant provisions from a
policy knowledge base, drops the ones below a relevance threshold, and assembles an acknowledgment
built **only** from the passages that survived, each one cited inline. When nothing clears the
threshold, it says so plainly and routes the matter to a licensed adjuster instead of inventing
coverage. No coverage determination is ever made at intake.

The grounding rule is structural rather than advisory: the answer text is assembled from retrieved
passage text, and the claimant's own question is never echoed into the document — so there is no
path by which caller text becomes response content. Both the grounded and the abstaining behaviour
are asserted end to end in the test suite.

Everything a caller can send is bounded. Identifiers are restricted to an inert alphabet because
they are printed into the acknowledgment; the question is length-capped; unknown fields are refused
rather than ignored; and caller text is screened for chat-template control tokens and explicit
instruction-override directives by the node that owns the contract, so refusal does not depend on a
gate configured elsewhere. The assembled document then passes an output boundary that withholds it
entirely — and clears every intermediate that carries it — if it finds credential material.

The bundled knowledge base is a small set of property-and-casualty provisions (comprehensive and
collision auto, full-glass endorsement, dwelling fire, personal accident, plus intake procedure and
general exclusions). Replacing it with your own corpus, or with a vector store, needs no change
outside the retrieval node.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent fails at
graph compile / start-up preflight rather than starting in a partially working state. This is
intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Calling it

`POST /invoke`, with a Bearer token when `INVOKE_AUTH_TOKEN` is set on the server:

```jsonc
{
  "input": "{\"claim_id\": \"INS-FNOL-20260712-001\", \"claim_type\": \"auto\", \"policy_number\": \"PC-AUTO-88231\", \"query\": \"My windshield was cracked by road debris — is the glass damage covered under my comprehensive auto policy?\"}",
  "input_context": { "channel": "claims_portal" }
}
```

- `claim_id` is required and must match `[A-Za-z0-9_-]{1,64}`; so must `policy_number` and
  `claim_type` when present.
- `query` is required, non-empty, and at most 2000 characters. It is the only free-text field.
- `input_context.channel` is optional and must match `[a-z0-9_]{1,32}`. It is recorded on the
  acknowledgment.
- Unknown fields are refused; unknown `input_context` keys are dropped at the entry point.

## Project Structure

```
src/graph/      outer backbone graph, inner retrieval pipeline, and the context bridge
src/nodes/      validation, retrieval, reranking, answer assembly, output boundary
src/services/   shared caller-input validation used by the nodes and the entry point
src/schemas/    the flat State definition and its serialization helpers
tests/          unit and boundary tests, including end-to-end through the real entry point
config/         agent manifest (registration) and runtime parameters
prompts/        the grounding contract used by the answer step
docs/           design and test specification
```

See `docs/` for the design specification and the test specification.

## Customising

1. Replace `_POLICY_KB` in `src/nodes/retrieve_node.py` with your own provisions, keeping the
   passage shape `{doc_id, source, policy_section, claim_types, text}` — or swap the scoring
   function for a vector store.
2. Tune `rag.top_k` and `rag.score_threshold` in `config/config.yaml`. Both are validated against
   documented ranges; an out-of-range or non-finite value is dropped rather than applied.
3. Adjust the acknowledgment wording in `src/nodes/output_format_node.py`, keeping the rule that
   only template text, knowledge-base text and inert identifiers are rendered.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
