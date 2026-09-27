import os
import tempfile
import unittest
from pathlib import Path

from src.prediction import (
    DEFAULT_MODELS_DIR,
    ModelLoadError,
    PredictionService,
)


class PredictionServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = PredictionService()

    def test_both_persisted_models_load(self):
        self.assertTrue(self.service.department_model_path.is_file())
        self.assertTrue(self.service.priority_model_path.is_file())
        self.assertTrue(hasattr(self.service._department_model, "predict"))
        self.assertTrue(hasattr(self.service._priority_model, "predict"))

    def test_valid_body_returns_both_labels(self):
        result = self.service.predict(
            "The office VPN keeps disconnecting and I cannot access the internal portal."
        )
        self.assertEqual(set(result), {"department", "priority"})
        self.assertIn(result["department"], self.service._department_model.classes_)
        self.assertIn(result["priority"], self.service._priority_model.classes_)

    def test_empty_and_whitespace_bodies_are_rejected(self):
        for body in ("", "   ", "\n\t"):
            with self.subTest(body=repr(body)):
                with self.assertRaisesRegex(ValueError, "empty or whitespace-only"):
                    self.service.predict(body)

    def test_non_string_input_is_rejected(self):
        for body in (None, 12, ["ticket"]):
            with self.subTest(body=repr(body)):
                with self.assertRaisesRegex(TypeError, "must be a string"):
                    self.service.predict(body)

    def test_prediction_is_deterministic(self):
        body = "The office VPN keeps disconnecting and I cannot access the internal portal."
        self.assertEqual(self.service.predict(body), self.service.predict(body))

    def test_default_model_paths_do_not_depend_on_current_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                os.chdir(temp_dir)
                service = PredictionService()
            finally:
                os.chdir(str(original))
        self.assertEqual(service.department_model_path,
                         (DEFAULT_MODELS_DIR / "department_classifier.joblib").resolve())
        self.assertEqual(service.priority_model_path,
                         (DEFAULT_MODELS_DIR / "priority_classifier.joblib").resolve())

    def test_missing_artifact_has_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ModelLoadError, "was not found"):
                PredictionService(
                    department_model_path=Path(temp_dir) / "missing.joblib",
                    priority_model_path=self.service.priority_model_path,
                )


if __name__ == "__main__":
    unittest.main()
