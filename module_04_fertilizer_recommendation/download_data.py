"""Download Dataset A for fertilizer recommendation.

Dataset B (the Kaggle playground synthetic table) is intentionally not
downloaded. The 15 October assignment says the two files stay separate.
"""

from pathlib import Path
from urllib.request import urlretrieve

URL = (
    "https://huggingface.co/spaces/SamarthPujari/"
    "Crop-and-Fertilizer-Recommendation-System/resolve/main/Datasets/"
    "Fertilizer%20Prediction.csv"
)
DESTINATION = Path(__file__).resolve().parent / "data" / "raw" / "fertilizer_prediction.csv"


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    urlretrieve(URL, DESTINATION)
    print(f"Saved {DESTINATION}")


if __name__ == "__main__":
    main()
