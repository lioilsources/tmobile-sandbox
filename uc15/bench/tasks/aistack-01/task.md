# Audio vibe: fail fast with 503 when the music model is down

`services/audio` (FastAPI). Overnight the music model container is stopped; vibe analysis then "succeeds"
with an empty caption, gets cached with the sample, and generation fails with an unreadable error.

- `app/backends/base.py`: add `MODEL_DOWN` (a human-readable message) and
  `class BackendUnavailable(BackendError)` whose message is always `MODEL_DOWN` (keep the technical
  detail in `.detail`); export it from `app.backends`.
- `AceStepBackend`: connection errors (`httpx.ConnectError`, `httpx.ConnectTimeout`) in generate and
  analyze raise `BackendUnavailable`.
- `POST /v1/audio/vibe/analyze` and `/v1/audio/vibe/generate`: check the music backend's `health()` first;
  if it is not ok → 503 with `detail == MODEL_DOWN` (sample upload still works without the model).
- Job runner: on `BackendUnavailable` do not try further variations (job error = `MODEL_DOWN`).
- Vibe analysis: `BackendUnavailable` from the LM is re-raised (job fails with `MODEL_DOWN`) instead of
  degrading to a librosa-only analysis; an analysis whose caption did not come from the LM is not cached
  with the sample.

Verify: `cd services/audio && python -m pytest tests/test_vibe.py`.
