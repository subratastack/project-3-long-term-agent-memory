# ADR-006: Memory Trust and Poisoning Defenses

- **Status:** Accepted
- **Date:** 2026-09-24

## Context

Agent inputs may contain hallucinations, malicious instructions, stale claims,
or prompt injection embedded in tool output. Persisting them can influence many
future sessions. Model confidence is not evidence, and semantic similarity is
not trust.

## Decision

Every event and candidate records source type, stable source identity, tenant,
integrity metadata when available, confidence, and provenance. A deterministic,
versioned write policy combines source trust, corroboration, candidate type,
content risk, temporal/conflict checks, and authorization to choose `accept`,
`supersede`, `quarantine`, or `reject`.

Safety-policy changes, secrets found in untrusted content, prompt-injection
instructions, and cross-tenant claims cannot become active memory through
automatic extraction. Procedural memory requires repeated trusted successful
episodes or explicit authorized approval. Quarantined records are isolated from
normal retrieval and require recorded review to change status.

LLM output is always an untrusted proposal. The LLM cannot assign its own source
authority, tenant, write decision, or approval.

## Consequences

- Poisoning risk is controlled before data enters normal retrieval.
- Every decision can be explained against provenance and policy version.
- Some useful but weakly supported memories are delayed or rejected.
- Policy maintenance, review tooling, and adversarial evaluation are required.

## Rejected alternatives

- **Trust model confidence:** confidence does not establish source integrity.
- **Store first, filter at retrieval:** poisoned content can leak into indexes,
  consolidation, or later pipelines.
- **Trust all tool output:** tools can return attacker-controlled content.
- **Use a single global trust score:** hides source, type, and policy-specific
  reasons needed for a decision.

## Validation

Measure acceptance, quarantine, and rejection for malicious safety overrides,
hallucinated model claims, tool-output prompt injection, low-trust external
facts, stale facts presented as current, and cross-tenant content. The target
for cross-tenant retrieval is zero.
