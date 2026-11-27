import argparse
from pathlib import Path

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def main():
    parser = argparse.ArgumentParser(description="Convert a scikit-learn pipeline to ONNX format.")
    parser.add_argument("--input", type=Path, required=True, help="Path to a joblib model artifact.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "crop_rf.onnx",
        help="Path for the ONNX model.",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input model not found: {args.input}")

    artifact = joblib.load(args.input)
    model = artifact.get("model")
    features = artifact.get("features")
    if model is None or not features:
        parser.error("Input artifact must contain a scikit-learn model and its feature list.")

    onnx_model = convert_sklearn(
        model,
        initial_types=[("features", FloatTensorType([None, len(features)]))],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(onnx_model.SerializeToString())
    print(f"ONNX model saved to: {args.output}")


if __name__ == "__main__":
    main()
