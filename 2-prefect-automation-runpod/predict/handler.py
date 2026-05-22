import os
import pickle

import runpod


def load_model():
    """Load trained model from local model directory."""
    model_path = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


MODEL = load_model()

FEATURES = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
    "PetalAreacm2",
    "SepalAreacm2",
]


def handler(event):
    """RunPod serverless handler function.

    Expected input: {"input": {"samples": [{"SepalLengthCm": 5.1, "SepalWidthCm": 3.5,
        "PetalLengthCm": 1.4, "PetalWidthCm": 0.2, "PetalAreacm2": 0.28,
        "SepalAreacm2": 17.85}]}}
    Returns: {"prediction": [0]}
    """
    input_data = event.get("input", {})
    samples = input_data.get("samples", [])

    if MODEL is None:
        return {"error": "Model not loaded"}

    if not samples:
        return {"error": "No samples provided"}

    for i, sample in enumerate(samples):
        missing = [f for f in FEATURES if f not in sample]
        if missing:
            return {"error": f"Sample {i} missing features: {missing}"}

    X = [[sample[f] for f in FEATURES] for sample in samples]
    predictions = MODEL.predict(X).tolist()
    return {"prediction": predictions}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
