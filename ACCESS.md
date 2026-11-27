# Machine-independent access (any laptop)

Same Cursor account on another machine: clone this repo, checkout `pilot`.

## Code + pilots + meetings (source of truth)
```bash
git clone https://github.com/AryanVashisth/VRPS.git
cd VRPS
git checkout pilot
```

| Path | Contents |
| --- | --- |
| `pilot/` | All 6 ONNX modules + dashboard + intern manual + validation audit |
| `meetings/` | Nov–Dec meeting folders (pptx + handouts) |
| `Documentation/` | Shared PDFs/DOCX + manuals |
| `docs_root/` | Intern guide PDF, session tracker |
| `ACCESS.md` | This file |

## Quick download of meeting/docs zip
https://github.com/AryanVashisth/VRPS/releases/tag/meetings-docs-2026-11-27

## Google Drive
Target folder: https://drive.google.com/drive/u/3/folders/10BdxeVioWVp5AVuJl0Pz9RHdyYRRvK02  
(`aryanvashisth01@gmail.com`)

Upload script (run once after Google login on that machine):
```bash
# from a machine where Chrome/Drive is signed in as aryanvashisth01@gmail.com
rclone config   # create remote name: gdrive, type: drive
rclone copy /path/to/VRPS/meetings gdrive:VRPS-Research-Associate/meetings -P
rclone copy /path/to/VRPS/Documentation gdrive:VRPS-Research-Associate/Documentation -P
```

Or drag the release zip into the Drive folder in the browser.

## Pilot console
```bash
cd pilot/dashboard
python3 -m venv ../.venv && source ../.venv/bin/activate
pip install -r ../module_01_soil_fertility/requirements.txt -r requirements.txt
uvicorn app:app --port 8765
```
