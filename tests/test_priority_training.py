import tempfile
import unittest
from pathlib import Path

import joblib

from src.preprocessing import normalized_body_key
from src.train_department import (
    build_candidates,
    duplicate_aware_split,
    prepare_department_rows,
)
from src.train_priority import (
    _reuse_department_split,
    prepare_priority_rows,
    run_experiment,
)


class PriorityTrainingTests(unittest.TestCase):
    def _rows(self):
        rows = [{"Body": "  ", "Department": "Technical", "Priority": "high",
                 "Tags": "excluded"}]
        for priority, prefix in (("high", "urgent"), ("medium", "normal"),
                                 ("low", "routine")):
            for number in range(5):
                body = "{0} ticket issue group{1}".format(prefix, number)
                rows.append({"Body": body, "Department": priority,
                             "Priority": priority, "Tags": "excluded"})
                rows.append({"Body": "  " + body.upper() + "  ",
                             "Department": priority, "Priority": priority,
                             "Tags": "excluded"})
        return rows

    def test_reuses_same_reproducible_duplicate_safe_partitions(self):
        rows = self._rows()
        texts, priorities, groups = prepare_priority_rows(rows)
        department_texts, departments, department_groups = prepare_department_rows(rows)
        expected = duplicate_aware_split(department_texts, departments,
                                         department_groups, random_seed=42)
        actual = _reuse_department_split(rows, texts, groups)
        self.assertEqual(expected, actual)
        self.assertEqual(len(texts), 30)
        self.assertEqual(len(priorities), 30)
        self.assertEqual(len(set(groups)), 15)
        group_sets = [{groups[index] for index in actual[name]}
                      for name in ("train", "validation", "test")]
        self.assertEqual(sum(bool(a & b) for i, a in enumerate(group_sets)
                             for b in group_sets[i + 1:]), 0)
        self.assertEqual(normalized_body_key(" urgent  ticket "),
                         normalized_body_key("URGENT ticket"))

    def test_candidate_tfidf_is_fitted_only_on_given_training_text(self):
        candidate = build_candidates()["Multinomial Naive Bayes"]
        candidate.fit(["alpha trainvocabulary", "bravo trainvocabulary",
                       "charlie anothertrainingword"],
                      ["high", "medium", "low"])
        vocabulary = candidate.named_steps["tfidf"].vocabulary_
        self.assertIn("trainvocabulary", vocabulary)
        self.assertNotIn("validationonlytoken", vocabulary)

    def test_training_saves_reloadable_priority_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "priority.joblib"
            report_path = Path(directory) / "report.md"
            result = run_experiment(self._rows(), model_path, report_path)
            self.assertTrue(model_path.is_file())
            self.assertTrue(report_path.is_file())
            self.assertIn(result["selected_model"], result["validation"])
            self.assertEqual(len(result["test"]["confusion_matrix"]), 3)
            self.assertEqual(len(joblib.load(str(model_path)).predict(
                ["urgent ticket issue"])), 1)


if __name__ == "__main__":
    unittest.main()
