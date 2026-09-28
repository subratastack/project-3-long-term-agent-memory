"""Domain enumerations for the Long-Term Agent Memory OS.

Think of these as the fixed vocabulary the rest of the system talks in: every
memory has a type, a lifecycle status, a trust tier, and so on, and those
values always come from one of the enums below. Keeping them centralized here
means a rule like "quarantined memories are never retrieved" only has to be
written once and can rely on `MemoryStatus.QUARANTINED` meaning the same thing
everywhere -- in the domain models, the database columns, and the write
policy.

All of these are `StrEnum`, so a member behaves like a plain string at
runtime (`MemoryStatus.ACTIVE == "active"` is `True`). That is what lets the
same value flow straight into a Pydantic field, a SQL `CHECK` constraint, and
a JSON API response without any manual conversion step.
"""

from enum import StrEnum


class MemoryType(StrEnum):
    """The cognitive category of a memory, and how long it is expected to last.

    - EPISODIC: something that happened -- a concrete observation, tool call,
      or session event tied to a specific moment in time.
    - SEMANTIC: a distilled fact, preference, or piece of world knowledge,
      usually built up from one or more episodic observations.
    - PROCEDURAL: a reusable routine or instruction for how the agent should
      act -- the highest-stakes type, since it shapes future behavior.
    """

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryStatus(StrEnum):
    """Where a memory currently sits in its lifecycle.

    This is what retrieval filters on before anything else: a memory that
    is not ACTIVE is invisible to ordinary reads, no matter how relevant its
    content might otherwise look.

    - ACTIVE: currently valid and eligible for retrieval.
    - QUARANTINED: held back pending review (e.g. low-trust evidence); not
      retrievable until promoted.
    - SUPERSEDED: replaced by a newer memory; kept for history and audit.
    - EXPIRED: past its `valid_to` timestamp.
    - TOMBSTONE: deliberately forgotten; the record is kept only so the
      deletion itself is auditable, not so the content can be recovered.
    """

    ACTIVE = "active"
    QUARANTINED = "quarantined"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    TOMBSTONE = "tombstone"


class TrustLevel(StrEnum):
    """How much a piece of content or its source should be trusted.

    Write policy and context-building both key off this to decide what an
    agent is allowed to see or act on. It exists specifically to blunt
    prompt-injection and memory-poisoning attempts: untrusted content can
    still be stored (for audit), but it never reaches an agent's context with
    the same authority as something trusted.

    - SYSTEM: the most reliable tier -- config, internal platform events.
    - HIGH: verified agent actions or trusted external tools.
    - MEDIUM: ordinary, unverified user or tool interaction.
    - LOW: weakly verified input.
    - UNTRUSTED: unverified third-party or suspicious content that should be
      isolated or quarantined rather than acted on.
    """

    UNTRUSTED = "untrusted"
    SYSTEM = "system"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SourceType(StrEnum):
    """Where a piece of evidence (a `MemoryEvent`) originally came from.

    This is the raw provenance signal that write policy uses to assign a
    baseline trust level -- e.g. a `SYSTEM_EVENT` starts out far more trusted
    than a `TOOL_OUTPUT`, before any further verification happens.

    - USER_MESSAGE: raw text typed by a human user.
    - TOOL_OUTPUT: whatever a called tool or external API returned.
    - SYSTEM_EVENT: an internal platform trigger or lifecycle notice.
    - AGENT_ACTION: something the agent itself planned or executed.
    - CONFIGURATION: an explicit, administrator-set policy or setting.
    """

    USER_MESSAGE = "user_message"
    TOOL_OUTPUT = "tool_output"
    SYSTEM_EVENT = "system_event"
    AGENT_ACTION = "agent_action"
    CONFIGURATION = "configuration"


class WriteDecision(StrEnum):
    """The outcome of running a memory candidate through the write policy.

    This is the one piece of vocabulary that ties the whole ingestion
    pipeline together: `write_policy.py` computes one of these values, and
    `service.py` decides what to persist based on it (see that module's
    "How it works" section for the exact mapping).

    - ACCEPT: store the candidate as a new active memory.
    - REJECT: discard the candidate, but still record *why* in the audit
      trail -- no memory row is created.
    - QUARANTINE: store the candidate, but flagged for review rather than
      immediately usable.
    - SUPERSEDE: accept the candidate while simultaneously retiring an
      existing memory it conflicts with or replaces.
    """

    ACCEPT = "accept"
    REJECT = "reject"
    QUARANTINE = "quarantine"
    SUPERSEDE = "supersede"


class IndexStatus(StrEnum):
    """Where a memory record stands with respect to its derived vector index.

    PostgreSQL is always the source of truth for a memory's *content and
    authorization*; this only tracks whether its embedding is in a fit state
    for `apps.memory_service.retrieval.semantic` to search against. A failed
    or pending embedding never hides or invalidates the memory itself -- it
    just means semantic search will not surface it until re-indexed.

    - PENDING: no embedding has been computed yet (the default for every
      newly written memory).
    - INDEXED: an embedding was computed and is safe to search against.
    - FAILED: embedding computation was attempted and did not succeed;
      distinguished from PENDING so a retry/reconciliation worker can tell
      "never tried" apart from "tried and needs another attempt".
    """

    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"


class ConflictType(StrEnum):
    """How two memories relate to each other in the memory graph.

    Used on `MemoryRelation` edges to record *why* one memory points at
    another, which is what lets supersession and conflict resolution be
    explained later rather than just silently happening.

    - CONTRADICTION: the two memories make incompatible factual claims.
    - DUPLICATE: the two memories say effectively the same thing.
    - SUPERSESSION: the newer memory replaces the older one.
    """

    CONTRADICTION = "contradiction"
    DUPLICATE = "duplicate"
    SUPERSESSION = "supersession"
