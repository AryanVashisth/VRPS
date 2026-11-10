"""Download Dataset A: crop yield for Indian states, 1997–2020.

The assignment lists a Kaggle dataset. This URL is a public mirror of that
crop_yield.csv. Dataset B (the FAO/Zenodo country table) is a different
collection and is not merged in.
"""

from pathlib import Path
from urllib.request import urlretrieve

URL = (
    "https://huggingface.co/datasets/jason1966/"
    "akshatgupta7_crop-yield-in-indian-states-dataset/resolve/main/crop_yield.csv"
)
DESTINATION = Path(__file__).resolve().parent / "data" / "raw" / "crop_yield.csv"


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    urlretrieve(URL, DESTINATION)
    print(f"Saved {DESTINATION}")


if __name__ == "__main__":
    main()
