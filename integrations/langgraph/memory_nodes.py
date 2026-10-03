"""The seven ordered stages of a memory-enabled LangGraph agent run."""

from __future__ import annotations

import logging
from collections.abc import Callable
from itertools import pairwise
from uuid import UUID

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from apps.memory_service.ingestion.candidate_extractor import CandidateExtractor
from apps.memory_service.ingestion.classifier import classify_memory_type
from apps.memory_service.ingestion.normalizer import normalize_candidate
from integrations.langgraph.memory_context import (
    AgentInput,
    AgentResult,
    MemoryGraphInput,
    MemoryGraphOutput,
    MemoryGraphState,
    MemoryRunContext,
    validate_packed_context,
)
from integrations.langgraph.store_adapter import MemoryStoreAdapter

logger = logging.getLogger(__name__)
ReasoningNode = Callable[[AgentInput], AgentResult]


class MemoryNodes:
    """Reusable nodes; only host lifecycle stages receive the store adapter."""

    def __init__(self, store: MemoryStoreAdapter, extractor: CandidateExtractor) -> None:
        self._store = store
        self._extractor = extractor

    def retrieve_memory(
        self, state: MemoryGraphState, runtime: Runtime[MemoryRunContext]
    ) -> MemoryGraphState:
        # Missing authenticated scope is a host programming error, not an outage.
        if not isinstance(runtime.context, MemoryRunContext):
            raise ValueError("MemoryRunContext is required from the authenticated host")
        update: MemoryGraphState = {
            "packed_context": None,
            "memory_context": "",
            "memory_token_count": 0,
            "retrieved_memory_ids": (),
            "candidates": (),
            "policy_decisions": (),
            "write_decisions": (),
            "feedback_event_ids": (),
            "memory_errors": (),
        }
        try:
            update["packed_context"] = self._store.retrieve(runtime.context, state["query_text"])
        except Exception as exc:
            logger.warning("Memory retrieval unavailable (%s)", type(exc).__name__)
            update["memory_errors"] = (f"retrieve_memory:{type(exc).__name__}",)
        return update

    def merge_memory_context(
        self, state: MemoryGraphState, runtime: Runtime[MemoryRunContext]
    ) -> MemoryGraphState:
        packed = state["packed_context"]
        text, tokens = "", 0
        memory_ids: tuple[UUID, ...] = ()
        errors = state["memory_errors"]
        if packed is not None:
            try:
                validate_packed_context(
                    packed,
                    runtime.context,
                    token_budget=self._store.token_budget,
                    count_tokens=self._store.count_tokens,
                )
                text = packed.text
                tokens = packed.token_count
                memory_ids = tuple(item.memory.memory_id for item in packed.memories)
            except ValueError:
                errors += ("merge_memory_context:invalid_pack",)
        return {
            "agent_input": AgentInput(state["query_text"], text),
            "memory_context": text,
            "memory_token_count": tokens,
            "retrieved_memory_ids": memory_ids,
            "memory_errors": errors,
            # Raw selected records do not need to survive the prompt boundary.
            "packed_context": None,
        }

    def extract_memory_candidates(
        self, state: MemoryGraphState, runtime: Runtime[MemoryRunContext]
    ) -> MemoryGraphState:
        try:
            events = self._store.load_outcomes(runtime.context, state["result"].outcome_event_ids)
            if not events:
                return {"candidates": ()}
            # This node runs once after the completed reasoning/tool loop.
            candidates = tuple(
                normalize_candidate(raw, classify_memory_type(raw, events), events)
                for raw in self._extractor.extract(events)
            )
            return {"candidates": candidates}
        except Exception as exc:
            # A retrieval outage may also make outcome evidence unavailable.
            # Preserve the completed run, but never learn without loaded evidence.
            logger.warning("Memory extraction unavailable (%s)", type(exc).__name__)
            return {
                "candidates": (),
                "memory_errors": state["memory_errors"]
                + (f"extract_memory_candidates:{type(exc).__name__}",),
            }

    def apply_memory_write_policy(
        self, state: MemoryGraphState, runtime: Runtime[MemoryRunContext]
    ) -> MemoryGraphState:
        return {
            "policy_decisions": tuple(
                self._store.evaluate(runtime.context, candidate)
                for candidate in state["candidates"]
            )
        }

    def persist_memories(
        self, state: MemoryGraphState, runtime: Runtime[MemoryRunContext]
    ) -> MemoryGraphState:
        if len(state["policy_decisions"]) != len(state["candidates"]):
            raise ValueError("every extraction must pass the policy stage before persistence")
        return {
            "write_decisions": tuple(
                self._store.persist(runtime.context, candidate) for candidate in state["candidates"]
            )
        }

    def record_memory_feedback(
        self, state: MemoryGraphState, runtime: Runtime[MemoryRunContext]
    ) -> MemoryGraphState:
        try:
            return {
                "feedback_event_ids": self._store.record_feedback(
                    runtime.context,
                    state["retrieved_memory_ids"],
                    state["result"].useful_memory_refs,
                )
            }
        except Exception as exc:
            logger.warning("Memory feedback unavailable (%s)", type(exc).__name__)
            return {
                "feedback_event_ids": (),
                "memory_errors": state["memory_errors"]
                + (f"record_memory_feedback:{type(exc).__name__}",),
            }


def build_memory_graph(
    store: MemoryStoreAdapter,
    extractor: CandidateExtractor,
    reasoning: ReasoningNode,
) -> CompiledStateGraph[MemoryGraphState, MemoryRunContext, MemoryGraphInput, MemoryGraphOutput]:
    """Wrap a complete Project 1-style reasoning/tool loop in memory stages.

    The callback receives only AgentInput and must return AgentResult when
    its run finishes. Tool wrappers record outcomes separately through the
    host adapter. Pass MemoryRunContext via graph.invoke(..., context=...).
    """
    nodes = MemoryNodes(store, extractor)

    def agent_reasoning(state: MemoryGraphState) -> MemoryGraphState:
        result = reasoning(state["agent_input"])
        if not isinstance(result, AgentResult):
            raise TypeError("reasoning must return a completed AgentResult")
        return {"result": result}

    graph = StateGraph(
        MemoryGraphState,
        context_schema=MemoryRunContext,
        input_schema=MemoryGraphInput,
        output_schema=MemoryGraphOutput,
    )
    graph.add_node("retrieve_memory", nodes.retrieve_memory)
    graph.add_node("merge_memory_context", nodes.merge_memory_context)
    graph.add_node("agent_reasoning", agent_reasoning)
    graph.add_node("extract_memory_candidates", nodes.extract_memory_candidates)
    graph.add_node("apply_memory_write_policy", nodes.apply_memory_write_policy)
    graph.add_node("persist_memories", nodes.persist_memories)
    graph.add_node("record_memory_feedback", nodes.record_memory_feedback)
    stages = (
        "retrieve_memory",
        "merge_memory_context",
        "agent_reasoning",
        "extract_memory_candidates",
        "apply_memory_write_policy",
        "persist_memories",
        "record_memory_feedback",
    )
    graph.add_edge(START, stages[0])
    for before, after in pairwise(stages):
        graph.add_edge(before, after)
    graph.add_edge(stages[-1], END)
    return graph.compile()
