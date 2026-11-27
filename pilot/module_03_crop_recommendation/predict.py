"""Terminal crop recommendation using ONNX artifacts (VRPS-aligned)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.infer import CropRecommendationModel

FEATURES = ["N", "P", "K", "pH", "Temperature", "Humidity", "Rainfall"]
DEFAULT_ART = Path(__file__).resolve().parent / "artifacts"


def get_feature_value(value, prompt, parser):
    if value is not None:
        return value
    try:
        return float(input(prompt))
    except ValueError:
        parser.error(f"{prompt.rstrip(': ')} must be a number.")


def main():
    parser = argparse.ArgumentParser(
        description="Recommend a crop from soil and weather measurements using an ONNX model."
    )
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ART, help="Artifacts directory.")
    parser.add_argument("--n", type=float, help="Nitrogen value")
    parser.add_argument("--p", type=float, help="Phosphorus value")
    parser.add_argument("--k", type=float, help="Potassium value")
    parser.add_argument("--ph", type=float, help="Soil pH")
    parser.add_argument("--temperature", type=float, help="Temperature")
    parser.add_argument("--humidity", type=float, help="Humidity")
    parser.add_argument("--rainfall", type=float, help="Rainfall")
    args = parser.parse_args()

    if not (args.artifacts / "schema.json").is_file():
        parser.error(f"Artifacts not found in {args.artifacts}. Run: python -m src.train")

    values = {
        "N": get_feature_value(args.n, "Nitrogen (N): ", parser),
        "P": get_feature_value(args.p, "Phosphorus (P): ", parser),
        "K": get_feature_value(args.k, "Potassium (K): ", parser),
        "pH": get_feature_value(args.ph, "Soil pH: ", parser),
        "Temperature": get_feature_value(args.temperature, "Temperature: ", parser),
        "Humidity": get_feature_value(args.humidity, "Humidity: ", parser),
        "Rainfall": get_feature_value(args.rainfall, "Rainfall: ", parser),
    }

    result = CropRecommendationModel(args.artifacts).predict_one(values)
    print(f"Recommended crop: {result['crop']}")
    print(f"Model score: {float(result['confidence']):.1%}")
    print("Other predictions:")
    shown = 0
    for item in result["top3"]:
        if item["crop"] != result["crop"]:
            print(f"  {item['crop']}: {float(item['probability']):.1%}")
            shown += 1
            if shown == 2:
                break


if __name__ == "__main__":
    main()
