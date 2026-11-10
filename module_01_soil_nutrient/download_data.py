"""Download the soil-fertility table used by Module 1.

The 15 October 2026 assignment points at a Kaggle notebook, not a raw file.
That notebook's author publishes the original table here. This is the raw
file, not the modified file in which extra rows were added for the rare class.
"""

from pathlib import Path
from urllib.request import urlretrieve

URL = (
    "https://raw.githubusercontent.com/iurwpoietknckvjndfsm-gndvkd/"
    "ML---Soil-Quality/main/Data/Raw%20Data.csv"
)
DESTINATION = Path(__file__).resolve().parent / "data" / "raw" / "soil_fertility.csv"


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    urlretrieve(URL, DESTINATION)
    print(f"Saved {DESTINATION}")


if __name__ == "__main__":
    main()
