"""Run one example Body through both saved ticket classifiers.

The example demonstrates the interface only; its predicted labels have no
known ground-truth annotation and are not an accuracy claim.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction import PredictionService


EXAMPLE_BODY = (
    "The office VPN keeps disconnecting and I cannot access the internal portal. "
    "This is preventing me from submitting today's work."
)


if __name__ == "__main__":
    service = PredictionService()
    print(json.dumps({
        "body": EXAMPLE_BODY,
        "prediction": service.predict(EXAMPLE_BODY),
        "ground_truth": None,
    }, indent=2))
