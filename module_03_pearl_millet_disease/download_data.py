"""Download the three small Zenodo leaf archives and unpack them.

Smut is about 230 MB and Pyricularia is about 370 MB. Those two are left
out so this module stays small enough for a normal PC. The zip is deleted
after it is unpacked.
"""

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
IMAGES = ROOT / "data" / "images"

ARCHIVES = {
    "healthy": (
        "https://zenodo.org/api/records/17086194/files/PESGL_healthy%20leaves.zip/content",
        "healthy.zip",
    ),
    "sclerospora_graminicola": (
        "https://zenodo.org/api/records/17086194/files/PESGL_SCLPGR_Sclerospora%20graminicola.zip/content",
        "sclerospora.zip",
    ),
    "exserohilum_rostratum": (
        "https://zenodo.org/api/records/17086194/files/PESGL_DRECRO_exserohilum%20rostratum.zip/content",
        "exserohilum.zip",
    ),
}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for class_name, (url, filename) in ARCHIVES.items():
        destination = IMAGES / class_name
        destination.mkdir(parents=True, exist_ok=True)
        zip_path = RAW / filename
        print(f"Downloading {class_name}")
        urlretrieve(url, zip_path)
        print(f"Extracting {class_name}")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(destination)
        zip_path.unlink()
    print(f"Images are under {IMAGES}")


if __name__ == "__main__":
    main()
