# Troubleshooting

## Existing React test cannot find “Choose an image”

The upload uses a visually hidden file input and a separate Browse files button. The input now has `aria-label="Choose an image"` so it remains accessible and queryable in React Testing Library. Keep test queries aligned with accessible names rather than DOM layout.

## Docker build fails at `npm ci`

Ensure `frontend/package-lock.json` exists and matches `frontend/package.json`. Run `npm install` locally to update the lockfile, review the diff, then rebuild.

## Frontend loads but API requests fail

1. `docker compose ... ps` — verify both `api` and `frontend` are healthy/running.
2. `docker compose ... logs --tail=100 api frontend` — inspect startup/proxy errors.
3. Check `NEOJAUNDICE_API_UPSTREAM` and ensure frontend and API share a Compose network.
4. Verify API health from the host at `http://127.0.0.1:8001/api/v1/health`.
5. Check browser Network tab for the failing `/api/v1/predict` response.

## `class 0/1` appears or label is not trustworthy

The UI now renders provisional readable labels. The source target is `treatment_label`; verify its codebook and source documentation before assigning “jaundice” semantics. Confirm the probability definition in the backend inference code and response schema.

## Model loads but prediction seems implausible

Verify the exact checkpoint paths mounted at `/app/checkpoints`, preprocessing parity between training and inference, expected input dimensions/normalization, regression target units, and model version/hash. API health alone does not establish inference correctness.

## Dark-mode text is invisible

Add explicit `dark:text-*`, `dark:bg-*`, and `dark:border-*` classes to new result cards. Avoid relying only on inherited text colors or hard-coded light-only backgrounds.
