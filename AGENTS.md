# AGENTS.md

## Project scope

This repository is a multimodal event extraction agent built around a LangGraph pipeline.
The current graph is:

perception -> local_rag -> search -> fusion -> extraction -> verifier -> repair

Your job is to improve the RAG subsystem and its offline build pipeline without breaking the current graph contract.

## Repository map

- `mm_event_agent/graph.py`
  - Owns the graph wiring and node order.
  - Do not change node names or graph order unless the task explicitly asks for it.

- `mm_event_agent/layered_rag.py`
  - Current local RAG compatibility layer.
  - Existing exported entry points such as `build_index(...)` and `retrieve(...)` should remain backward compatible unless the task explicitly allows a breaking change.

- `mm_event_agent/schemas.py`
  - Owns downstream data contracts.
  - `TextEventExample`, `ImageSemanticExample`, `BridgeExample`, and `LayeredSimilarEvents` are important compatibility surfaces.

- `mm_event_agent/ontology.py`
  - This is the canonical ontology sink.
  - New dataset labels from ACE2005, MAVEN-ARG, and SWiG should map into this ontology instead of creating a parallel ontology.

- `mm_event_agent/runtime_config.py` and `.env.example`
  - Runtime config and env defaults must stay in sync.
  - When adding a new environment variable, update both.

- `tests/test_smoke.py`
  - Existing smoke/integration baseline.
  - Do not break it silently.

## Current architectural target

The repository currently has a demo-style in-memory layered RAG.
The target is a persistent layered RAG with offline-built indexes.

Target retrieval layout:

- Text branches:
  - ACE2005 text exemplars
  - MAVEN-ARG text exemplars
  - SWiG text/semantic exemplars
  - bridge exemplars

- Image branch:
  - SWiG image exemplars encoded with Qwen3-VL-Embedding

Important:
- Text retrieval and image retrieval are different branches.
- Do not force everything into one shared encoder.
- Preserve the existing three output layers:
  - `text_event_examples`
  - `image_semantic_examples`
  - `bridge_examples`

## Data and file placement conventions

If you add offline build tools or persistent artifacts, use this layout:

- `mm_event_agent/rag/`
  - reusable RAG modules
- `scripts/`
  - offline build scripts
- `data/rag/normalized/`
  - normalized JSONL corpora
- `data/rag/embeddings/`
  - intermediate embedding shards
- `data/rag/indexes/`
  - persistent FAISS indexes and metadata

Each persistent index directory should eventually contain:
- `index.faiss`
- `meta.jsonl`
- `build_info.json`

## Rules for modifications

- Prefer additive, backward-compatible changes.
- Keep diffs small and reviewable.
- Keep public interfaces stable unless the task explicitly permits a breaking change.
- Do not rewrite extraction, verifier, or repair prompting logic during RAG refactors unless explicitly asked.
- Do not rename top-level state keys casually.
- Do not claim tests passed unless you actually ran them.
- If a dependency or dataset is missing, say exactly what could not be run.

## RAG-specific rules

- Use dense text embeddings for text branches.
- Use Qwen3-VL image embeddings only for the SWiG image branch.
- Keep text and image metadata rich enough for later reranking and citation.
- Canonical event type and role mapping must flow into `ontology.py`-compatible labels.
- Favor explicit metadata fields over ad-hoc free-form strings.
- When adding retrieval metadata, do not remove the existing fields that downstream nodes already consume.

## Coding style

- Use Python type hints.
- Prefer small classes or pure functions over large monolithic scripts.
- For JSONL corpora, keep one record per line and use UTF-8.
- For CLI scripts, use `argparse`.
- For offline scripts, make outputs deterministic and document file naming clearly.
- Avoid hidden global state unless it is already part of the compatibility layer.

## Validation

For code that touches the current runtime path:
- Run `python -m unittest tests.test_smoke` when possible.

For code that only adds offline builders or new RAG modules:
- Add focused tests under `tests/` when practical.
- At minimum, verify imports and basic construction paths.

In the final response for each task, include:
1. files changed
2. why each file changed
3. what commands were run
4. what could not be validated
5. obvious next step, if any

## Working style

For non-trivial tasks:
- First read the relevant files.
- Then write a short plan with the files you intend to modify.
- Then implement.
- Then run the most relevant checks.
- Then summarize the result clearly.

When uncertain:
- preserve compatibility first
- document assumptions in code comments only when necessary
- avoid inventing behavior that was not requested