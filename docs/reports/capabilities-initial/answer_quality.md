# Answer quality with MemoryAgentBench evidence

Retrieval can find useful text while the agent still answers incorrectly. This experiment asks the same six MemoryAgentBench questions with no supplied memory and with the earlier reranked top-five chunks packed into a 2,000-token memory budget. The instruction requires abstention when the supplied text cannot support an answer.

These are measured results from 2026-10-04T00:17:01.371235+00:00. The JSON companion contains settings, fixture hashes, and individual outcomes.

## What the metrics mean

Exact match compares the complete normalized final answer with a reference alias. Alias containment checks a whole-word reference inside a longer answer. Token F1 measures word overlap. Evidence hit means a labeled relevant chunk survived packing; citation validity checks IDs only, not whether the cited text proves the answer. Invalid JSON scores zero and remains in the denominator.

## Measured results

| Source | Arm | Questions | Exact | Alias | Token F1 | Evidence hit | Citation IDs valid | Abstention | Valid JSON |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ruler_qa1_197K | no_memory | 2 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 1 | 1 |
| ruler_qa1_197K | retrieved_memory | 2 | 0.5000 | 0.5000 | 0.9286 | 1 | 1 | 0 | 1 |
| ruler_qa2_421K | no_memory | 2 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 1 | 1 |
| ruler_qa2_421K | retrieved_memory | 2 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 1 |
| longmemeval_s* | no_memory | 2 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 1 | 1 |
| longmemeval_s* | retrieved_memory | 2 | 0.0000 | 0.0000 | 0.0000 | 1 | 1 | 0 | 1 |

## What we learned

The no-memory arm abstained on all six questions, as instructed. With
retrieved memory, two of six answers matched the complete reference alias
exactly, and five of six had a labeled relevant chunk in their packed
evidence. All twelve calls returned valid answer JSON. These small counts
describe the pilot rather than a stable dataset-wide accuracy estimate.

The traces distinguish several causes of a low score:

- The Norse-origin answer lists Denmark, Iceland, and Norway, matching the
  reference's countries. It scores zero exact/alias match because the
  reference includes the word `and`, while the answer uses commas. Its token
  F1 is 0.8571. This is a scoring limitation, not a country-list error.
- The Latin-recording answer, `9th century`, and the age-comparison answer,
  `Terry Richardson`, match their references exactly.
- The Big Stone Gap question has no labeled relevant chunk after packing;
  the model abstains. Retrieval/evidence availability is the first issue to
  investigate for that question.
- Both LongMemEval answers disagree with their references: `45` rather than
  `50` work hours, and `one` rather than `Two` free nights. Their citation IDs
  exist and a labeled evidence chunk is present. Neither property proves
  that enough evidence was supplied or that the answer is supported. Review
  the full required evidence and packed chunks before assigning the failure
  entirely to generation.

The next answer-quality experiment should score equivalent lists and
paraphrases semantically and measure complete evidence coverage. A six-case
proxy result should not be reported as official MemoryAgentBench accuracy.

## Scope and next experiment

This is a six-question pilot, selected by trace order before generation, not the full dataset. Answer overlap is a diagnostic proxy, especially for LongMemEval prose, and is not MemoryAgentBench's official generated-answer score or an LLM judge. Answer-span relevance can overlabel common words. Next, enlarge the sample and add a separately calibrated semantic judge with human checks.

[Full results](answer_quality.json) · [Benchmark guide](../../capability-benchmarks.md)
