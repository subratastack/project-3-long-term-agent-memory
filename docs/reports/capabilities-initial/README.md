# Memory capability learning reports

Each report records one benchmark track, its scoring method, measured results, and limits. Model-backed tracks use installed local Ollama models. Policy, conflicts, lifecycle, and packing use the existing PostgreSQL database inside a transaction that is rolled back.

The first pilot used Qwen2.5 7B Instruct for answers and extraction, Qwen3 14B
for the thinking comparison, and MiniLM embeddings for packing. The model
digests, Ollama version, generation settings, input hashes, implementation
fingerprint, and individual outcomes are in the JSON companions.

| Track | Markdown | JSON |
| --- | --- | --- |
| write policy | [Learning report](write_policy.md) | [Traces](write_policy.json) |
| conflict resolution | [Learning report](conflict_resolution.md) | [Traces](conflict_resolution.json) |
| forgetting | [Learning report](forgetting.md) | [Traces](forgetting.json) |
| prompt packing | [Learning report](prompt_packing.md) | [Traces](prompt_packing.json) |
| answer quality | [Learning report](answer_quality.md) | [Traces](answer_quality.json) |
| memory extraction | [Learning report](memory_extraction.md) | [Traces](memory_extraction.json) |
| reasoning | [Learning report](reasoning.md) | [Traces](reasoning.json) |

The strongest immediate learning is that useful retrieval and valid source
IDs do not guarantee a correct answer. Both LongMemEval pilot answers were
wrong despite having labeled evidence in the prompt. Extraction scores also
need a careful reading: a supported preference paraphrase and a combined
two-fact proposal were undercounted by the word-based scorer. The reasoning
fixture was too easy to separate the two thinking modes: both passed six of
six cases, with higher latency when thinking was enabled.

The deterministic regression tracks passed all labeled policy, conflict, and
forgetting checks. Packing improved useful-group coverage under a tight
budget and removed duplicate groups, while larger prompts still admitted
distractors. These are component diagnostics, with authored cases for six
tracks and an external MemoryAgentBench pilot for answers.

The [archived JSON-constraint diagnostic](reasoning-json-constraint-diagnostic.md)
records an initial local Ollama output-format failure. The main reasoning
report uses the corrected request format for both modes.

[Method and rerun instructions](../../capability-benchmarks.md)
