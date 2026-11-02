# Documentation and repository cleanup policy

## Single source of truth

- `README.md`: project overview and shortest supported quick start.
- `docs/README.md`: documentation index.
- `docs/architecture.md`: architecture.
- `docs/setup.md`: local development and validation.
- `docs/api.md`: API contract.
- `docs/ml_pipeline.md`: dataset/training/evaluation.
- `docs/deployment.md`: Docker and deployment.
- `docs/troubleshooting.md`: operational fixes.
- `docs/research.md`: research questions, protocol, limitations.
- `CHANGELOG.md`: dated changes only.

Root `ARCHITECTURE.md`, `PROJECT_BLUEPRINT.md`, and `RESEARCH_NOTES.md` should remain short summaries or point to their canonical document; do not maintain the same detailed plan in two places.

## Files that should not normally be committed

- `.venv/`, `node_modules/`, `frontend/dist/`, `coverage/`, `.pytest_cache/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`
- local `.env*` files (except safe `*.example` templates)
- raw newborn images, private metadata, large generated checkpoints, temporary logs, scratch archives, and downloaded datasets unless repository policy explicitly tracks them with DVC/LFS or external artifact storage.

## Before deleting anything

1. Run `git status --short` and make a backup/commit.
2. Search references to the candidate path (`git grep`).
3. Check whether scripts, tests, CI, Dockerfiles, or docs depend on it.
4. Compare duplicate docs and move unique facts into the canonical file.
5. Delete only after a clean test/build and a review of the diff.

`TRUTH.md` appears to be assistant/project guidance rather than user-facing technical documentation in the supplied material. If it is tracked in the repository, consider moving it outside the public repo or deleting it only after confirming it is not required by tooling. Do not delete it blindly.
