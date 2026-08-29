# Decisions log

Resolved product/architecture questions, in order asked. See ARCHITECTURE.md for how each plays out in code.

1. **ML inference location**: separate Python microservice (FastAPI), not inside NestJS. NestJS orchestrates over HTTP. *Why*: no viable Node-native path to faster-whisper/pyannote/llama.cpp at comparable accuracy.

2. **Model stack**: faster-whisper (distil-large-v3, int8) + Piper + pyannote.audio (speaker-diarization-3.1) + Qwen2.5-3B-Instruct GGUF Q4_K_M via llama-cpp-python, single LLM prompted for summary+sentiment+QA rather than 3 models. *Why*: fits ~10GB of a 16GB CPU-only budget; separate models per task would blow the budget.

3. **Repo structure**: NO monorepo tooling (no pnpm workspaces / turborepo / nx). Plain independent folders: `backend/`, `frontend/`, `ml-service/`, each with own package manager state. No shared-types workspace package — use OpenAPI-generated client instead. *Why*: user explicit preference; also 3 different runtimes anyway so a JS-only workspace tool wouldn't unify them all.

4. **TTS scope**: BOTH (a) pipeline stage reading the generated Summary aloud, and (b) a standalone free-text-to-speech tool in the Playground, unrelated to uploaded-audio analysis. *Why*: user wants both, not either/or.

5. **OpenCode CLI page**: external project — page links to the official upstream OpenCode CLI GitHub releases per platform. No backend hosting/build pipeline, no NestJS module needed for it. *Why*: user confirmed it's the third-party open-source project, IntelliSense doesn't own/ship it.

6. **Docker**: not used for local dev v1. *Why*: Docker Desktop's WSL2 backend defaults to capping the VM at ~50% of host RAM (8GB on this 16GB machine) — would break the ~10GB model-loading budget unless `.wslconfig` manually raised. Native installs are also easier to debug for torch/ctranslate2/llama-cpp-python build issues on Windows. Revisit later purely for *distribution* (docker-compose for other users), not dev.

7. **Package manager**: npm, independently in `backend/` and `frontend/` (no root install step, no workspace). *Why*: consistent with decision 3 — simplest option once workspaces are off the table.

8. **Python version**: 3.13 for `ml-service`, not the also-installed 3.14. *Why*: 3.14 is too new — no prebuilt CPU wheels yet for torch/ctranslate2/llama-cpp-python at time of scaffolding (2026-08-28). Re-check if bumping later.

9. **Job concurrency**: ml-service pins job-queue concurrency to 1 (never processes two audio jobs at once). *Why*: this — not lazy-loading — is what keeps the app under the 16GB ceiling; CPU inference activation memory can spike per job.

10. **Job history persistence**: dropped Prisma/SQLite. ml-service already writes a `job.json` sidecar per job under `storage/audio-jobs/{jobId}/` as the durability mechanism; NestJS's `AudioJobsService.history()` just lists that shared directory instead of maintaining a second copy of the same state in a DB. *Why*: avoids a duplicate-persistence sync bug class for a local single-user tool; Prisma 7's install was also unexpectedly heavy/slow on this machine and added nothing history-wise that the filesystem didn't already have.

11. **Frontend/backend type sharing**: dropped OpenAPI codegen for v1, hand-written fetch client in `frontend/src/api/client.ts` matching backend DTOs by hand instead. *Why*: time-boxed initial build; codegen wiring (`openapi-typescript` against Nest's `/api-json`) is still the intended end state per ARCHITECTURE.md, tracked as backlog, not abandoned.

## Open / not yet decided
- Exact QA rubric criteria (placeholder: Greeting, Active Listening, Empathy, Resolution, Compliance, Professionalism) — needs stakeholder input.
- Default STT language scope (currently English-tuned `distil-large-v3`; `large-v3-turbo` multilingual swap is a config change away, not yet triggered by anything).
- Job history persistence: planned Prisma+SQLite, not yet scaffolded — confirm still wanted when backend orchestration phase starts.
- Max audio length target (short clips vs full 30-60min calls) — pipeline is designed for map-reduce chunking on long transcripts but this isn't implemented yet.
