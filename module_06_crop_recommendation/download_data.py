"""Download the crop-recommendation table assigned on 15 October 2026.

Kaggle requires an account, so this uses a public copy of the same file:
2,200 rows and the seven fields described in the assignment.
"""

from pathlib import Path
from urllib.request import urlretrieve

URL = "https://raw.githubusercontent.com/saisiddharth14/crop-recommendation/main/Crop_recommendation.csv"
DESTINATION = Path(__file__).resolve().parent / "data" / "raw" / "crop_recommendation.csv"


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    urlretrieve(URL, DESTINATION)
    print(f"Saved {DESTINATION}")


if __name__ == "__main__":
    main()
