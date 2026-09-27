import tempfile
import unittest
from pathlib import Path

import joblib

from src.preprocessing import normalized_body_key
from src.train_department import (
    build_candidates,
    duplicate_aware_split,
    prepare_department_rows,
    run_experiment,
)


class DepartmentTrainingTests(unittest.TestCase):
    def _rows(self):
        rows = [{"Body": "   ", "Department": "Class A", "Tags": "leak"}]
        for label, prefix in (("Class A", "alpha"), ("Class B", "bravo"),
                              ("Class C", "charlie")):
            for number in range(5):
                body = "{0} ticket category {1}".format(prefix, number)
                rows.append({"Body": body, "Department": label, "Tags": "excluded"})
                rows.append({"Body": "  " + body.upper() + "  ",
                             "Department": label, "Tags": "excluded"})
        return rows

    def test_split_is_reproducible_and_duplicate_groups_do_not_cross(self):
        texts, labels, groups = prepare_department_rows(self._rows())
        first = duplicate_aware_split(texts, labels, groups, random_seed=19)
        second = duplicate_aware_split(texts, labels, groups, random_seed=19)
        self.assertEqual(first, second)
        group_sets = [{groups[index] for index in first[name]}
                      for name in ("train", "validation", "test")]
        self.assertFalse(group_sets[0] & group_sets[1])
        self.assertFalse(group_sets[0] & group_sets[2])
        self.assertFalse(group_sets[1] & group_sets[2])
        self.assertEqual(len(texts), 30)  # blank Body was omitted
        self.assertEqual(len(set(groups)), 15)  # normalized duplicates are grouped
        self.assertEqual(normalized_body_key(" alpha  beta "),
                         normalized_body_key("ALPHA beta"))

    def test_pipeline_learns_tfidf_only_from_fit_rows(self):
        model = build_candidates()["Multinomial Naive Bayes"]
        model.fit(["alpha trainingphrase", "bravo trainingphrase",
                   "charlie fitphrase"],
                  ["Class A", "Class B", "Class C"])
        vocabulary = model.named_steps["tfidf"].vocabulary_
        self.assertNotIn("validationonlytoken", vocabulary)
        self.assertIn("trainingphrase", vocabulary)

    def test_training_saves_reloadable_pipeline_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "department.joblib"
            report_path = Path(directory) / "report.md"
            result = run_experiment(self._rows(), model_path, report_path)
            self.assertTrue(model_path.is_file())
            self.assertTrue(report_path.is_file())
            self.assertIn(result["selected_model"], result["validation"])
            self.assertEqual(set(result["split_rows"]),
                             {"train", "validation", "test"})
            self.assertEqual(len(result["test"]["confusion_matrix"]), 3)
            reloaded = joblib.load(str(model_path))
            self.assertEqual(len(reloaded.predict(["alpha ticket category"])), 1)


if __name__ == "__main__":
    unittest.main()
