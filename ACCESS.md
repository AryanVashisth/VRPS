# Machine-independent access

## Code (any laptop with Cursor / git)
```bash
git clone https://github.com/AryanVashisth/VRPS.git
cd VRPS
git checkout pilot
```
Pilots live under `pilot/`. Intern manual: `pilot/INTERN_USER_MANUAL.md`.

## Meetings and presentation docs
Synced under `meetings/` (Nov–Dec packs) and `Documentation/` in this branch.
Also mirrored to Google Drive folder (VRPS RA shared folder).

## Pilot console
```bash
cd pilot/dashboard
python3 -m venv ../.venv && source ../.venv/bin/activate
pip install -r ../module_01_soil_fertility/requirements.txt -r requirements.txt
uvicorn app:app --port 8765
```
