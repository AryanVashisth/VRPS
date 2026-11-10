"""Download the Zenodo soil-water-content bundle and keep the sensor CSV.

The bundle also contains rainfall and irrigation workbooks. This baseline
forecasts soil water content from the sensor's own past readings. Those
workbooks are not merged in this version; see the module README.
"""

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

URL = "https://zenodo.org/api/records/20665025/files/data.zip/content"
ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / "data" / "raw" / "zenodo_data.zip"
EXTRACTED = ROOT / "data" / "raw" / "zenodo"
CSV_NAME = "z6-03235_port1-2_water_content.csv"
DESTINATION = ROOT / "data" / "raw" / "port1_port2_water_content.csv"


def main() -> None:
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"Downloading {URL}")
        urlretrieve(URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(EXTRACTED)
    matches = list(EXTRACTED.rglob(CSV_NAME))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {CSV_NAME}, found {matches}")
    DESTINATION.write_bytes(matches[0].read_bytes())
    print(f"Saved {DESTINATION}")


if __name__ == "__main__":
    main()
