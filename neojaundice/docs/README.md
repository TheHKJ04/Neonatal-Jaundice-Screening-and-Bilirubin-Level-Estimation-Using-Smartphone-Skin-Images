# Documentation index

This directory is the canonical source for project documentation. Keep product documentation here and link to it from the root README.

| Document | Purpose |
|---|---|
| [`setup.md`](setup.md) | Local setup, environment, commands, validation |
| [`architecture.md`](architecture.md) | Current architecture and boundaries |
| [`api.md`](api.md) | API routes, request/response contract, verification |
| [`ml_pipeline.md`](ml_pipeline.md) | Dataset, training, evaluation, experiment reproducibility |
| [`deployment.md`](deployment.md) | Docker frontend/API runbook and production notes |
| [`troubleshooting.md`](troubleshooting.md) | Known errors and debugging steps |
| [`research.md`](research.md) | Research questions, evaluation standards, limitations |
| [`cleanup.md`](cleanup.md) | Documentation ownership and repository hygiene |

## Status labels

- **Verified:** directly exercised by a passing test or command and recorded with evidence.
- **Implemented, not yet verified:** code/config exists, but no successful runtime/test evidence has been recorded.
- **Planned:** not implemented yet.
- **Research-only:** must not be presented as a clinically validated feature.

Do not change a status to “verified” merely because files exist.
