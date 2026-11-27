# VRPS Pilot Console

Local dummy dashboard that loads every pilot ONNX module and runs the same `infer.py` contracts the Android path will use.

## Run

```bash
cd pilot/dashboard
# reuse the pilot venv if present
../.venv/bin/pip install -r requirements.txt
../.venv/bin/uvicorn app:app --port 8765
```

Open http://127.0.0.1:8765

## What it is for

- Smoke-test Modules 1–6 from one screen
- Confirm schema field names before sensor mapping
- Demo the stack in meetings without claiming field accuracy

Predictions stay on-device style: local ONNX Runtime only.
