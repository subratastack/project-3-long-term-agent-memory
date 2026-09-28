# ADR-002: Memory Types and Promotion

- **Status:** Accepted
- **Date:** 2026-09-24

## Context

Events, short-lived reasoning state, durable facts, and reusable procedures
have different evidence and lifecycle requirements. Treating them as one pool
encourages transcript dumping, lets hallucinations become facts, and allows a
single untrusted instruction to shape future agent behavior.

## Decision

The system distinguishes working, episodic, semantic, and procedural memory.

- **Working memory** is runtime/graph state. It is not persisted as long-term
  memory merely because it appeared in a prompt or model response.
- **Episodic memory** records what happened and may be admitted from trusted
  execution events. It is normally immutable.
- **Semantic memory** records a durable claim supported by provenance,
  confidence, tenant scope, and conflict/validity checks.
- **Procedural memory** records reusable instructions. It requires repeated
  successful supporting episodes or explicit authorized approval and is
  versioned and deprecatable.

Promotion creates a new candidate and, after policy approval, a new memory
record. The promoted record links to every supporting event/memory. Original
episodes remain intact. No LLM extraction output promotes itself or writes
directly to the authoritative memory table.

Candidate classification is based on meaning and intended use:

| Candidate | Classification |
| --- | --- |
| A bounded occurrence, action, observation, or outcome | Episodic |
| A claim intended to remain true beyond one occurrence | Semantic |
| A reusable sequence or policy for future action | Procedural |
| A current hypothesis or pending task | Working; do not persist by default |

## Consequences

### Positive

- Admission rules and expiry behavior match the risk of each memory class.
- Evidence remains explainable after consolidation and promotion.
- Procedural poisoning has a deliberately high barrier.
- Retrieval can select only types appropriate to the current query.

### Negative

- Classification and promotion introduce extra policy and data relationships.
- Some candidates are ambiguous and must be quarantined or reviewed.
- Consolidation is eventually required to control repeated episodic evidence.

## Rejected alternatives

- **Store every interaction as episodic memory:** produces unbounded, noisy
  history and confuses evidence with useful memory.
- **Treat all extracted statements as semantic facts:** promotes hallucinations
  and stale assertions.
- **Infer procedures from one successful episode:** provides too little
  evidence for behavior-shaping instructions.
- **Mutate an episode into a fact/procedure:** destroys provenance and history.

## Validation

Tests must cover type boundaries, attempted promotion from insufficient or
untrusted evidence, preservation of supporting links, procedural versioning,
and the rule that working state is not automatically promoted.
