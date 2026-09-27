"""Application-facing prediction service for saved ticket classifiers."""

from pathlib import Path
from typing import Dict, Optional, Union

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


class ModelLoadError(RuntimeError):
    """Raised when a persisted classifier is unavailable or cannot be loaded."""


class PredictionError(RuntimeError):
    """Raised when a loaded classifier cannot produce a prediction."""


class PredictionService:
    """Load persisted text pipelines once and predict both ticket labels."""

    def __init__(
        self,
        department_model_path: Optional[Union[str, Path]] = None,
        priority_model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.department_model_path = Path(department_model_path or (
            DEFAULT_MODELS_DIR / "department_classifier.joblib"
        )).expanduser().resolve()
        self.priority_model_path = Path(priority_model_path or (
            DEFAULT_MODELS_DIR / "priority_classifier.joblib"
        )).expanduser().resolve()

        self._department_model = self._load_model(
            self.department_model_path, "Department"
        )
        self._priority_model = self._load_model(
            self.priority_model_path, "Priority"
        )

    @staticmethod
    def _load_model(path: Path, target_name: str):
        if not path.is_file():
            raise ModelLoadError(
                "The saved {0} model was not found at: {1}".format(target_name, path)
            )
        try:
            return joblib.load(str(path))
        except Exception as exc:
            raise ModelLoadError(
                "The saved {0} model could not be loaded from {1}: {2}".format(
                    target_name, path, exc
                )
            ) from exc

    def predict(self, body: str) -> Dict[str, str]:
        """Return Department and Priority predictions for one ticket Body.

        Raw text is passed to each persisted pipeline, which applies the exact
        preprocessing and TF-IDF logic learned during training. No confidence
        values are returned because the selected Linear SVMs do not provide
        calibrated class probabilities.
        """
        if not isinstance(body, str):
            raise TypeError("Ticket Body must be a string")
        if not body.strip():
            raise ValueError("Ticket Body cannot be empty or whitespace-only")

        try:
            department = self._department_model.predict([body])[0]
            priority = self._priority_model.predict([body])[0]
        except Exception as exc:
            raise PredictionError("Could not generate ticket predictions: {0}".format(exc)) from exc
        return {"department": str(department), "priority": str(priority)}
