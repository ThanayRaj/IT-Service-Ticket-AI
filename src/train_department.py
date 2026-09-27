"""Duplicate-aware Department model selection and final evaluation."""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import joblib

from src.data_loader import load_tickets
from src.preprocessing import clean_text, normalized_body_key


RANDOM_SEED = 42
SPLIT_FOLDS = 5
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "department_classifier.joblib"
REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "department_model_report.md"


def _clean_batch(texts: Sequence[str]) -> List[str]:
    return [clean_text(text) for text in texts]


def prepare_department_rows(rows: Sequence[dict]) -> Tuple[List[str], List[str], List[str]]:
    """Use only non-empty Body text and Department labels; return duplicate groups."""
    texts, labels, groups = [], [], []
    for row in rows:
        body = clean_text(row.get("Body"))
        label = str(row.get("Department", "")).strip()
        if body and label:
            texts.append(body)
            labels.append(label)
            groups.append(normalized_body_key(body))
    return texts, labels, groups


def duplicate_aware_split(
    texts: Sequence[str], labels: Sequence[str], groups: Sequence[str],
    random_seed: int = RANDOM_SEED,
) -> Dict[str, List[int]]:
    """Create 60/20/20 train/validation/test folds with groups kept intact.

    Five-fold StratifiedGroupKFold keeps repeated normalized bodies together
    while approximating the natural label mix at row level. Two distinct folds
    are reserved for validation and test; the other three form training.
    """
    if not (len(texts) == len(labels) == len(groups)):
        raise ValueError("texts, labels, and groups must have equal lengths")
    if len(set(labels)) < 2:
        raise ValueError("Department classification requires at least two classes")
    splitter = StratifiedGroupKFold(
        n_splits=SPLIT_FOLDS, shuffle=True, random_state=random_seed
    )
    folds = [test_index.tolist() for _, test_index in splitter.split(texts, labels, groups)]
    validation = folds[1]
    test = folds[0]
    train = [index for fold_number in range(2, SPLIT_FOLDS)
             for index in folds[fold_number]]
    result = {"train": train, "validation": validation, "test": test}
    _assert_group_isolation(result, groups)
    for name, indices in result.items():
        if not indices:
            raise ValueError("The {0} split is empty".format(name))
    return result


def _assert_group_isolation(splits: Dict[str, Sequence[int]], groups: Sequence[str]) -> None:
    names = list(splits)
    for i, first in enumerate(names):
        first_groups = {groups[index] for index in splits[first]}
        for second in names[i + 1:]:
            overlap = first_groups.intersection(groups[index] for index in splits[second])
            if overlap:
                raise ValueError("Duplicate body group crosses {0}/{1} splits".format(first, second))


def build_candidates() -> Dict[str, Pipeline]:
    """Return word TF-IDF pipelines with reasonable classical baselines."""
    candidates = {
        "Logistic Regression": LogisticRegression(
            C=2.0, max_iter=2000, class_weight=None, random_state=RANDOM_SEED
        ),
        "Linear SVM": LinearSVC(C=1.0, random_state=RANDOM_SEED),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
    }
    return {
        name: Pipeline([
            ("clean_text", FunctionTransformer(_clean_batch, validate=False)),
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_df=0.98,
                sublinear_tf=True, strip_accents="unicode", max_features=200000
            )),
            ("classifier", estimator),
        ])
        for name, estimator in candidates.items()
    }


def evaluate(estimator: Pipeline, texts: Sequence[str], labels: Sequence[str],
             class_names: Sequence[str]) -> Dict[str, object]:
    predictions = estimator.predict(texts)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=class_names, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=class_names, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=class_names).tolist(),
        "class_names": list(class_names),
    }


def _indexed(values: Sequence[str], indices: Sequence[int]) -> List[str]:
    return [values[index] for index in indices]


def _confusion_table(matrix: Sequence[Sequence[int]], names: Sequence[str]) -> str:
    lines = ["| Actual \\ Predicted | " + " | ".join(names) + " |",
             "|---|" + "---:|" * len(names)]
    for name, row in zip(names, matrix):
        lines.append("| {0} | {1} |".format(name, " | ".join(str(value) for value in row)))
    return "\n".join(lines)


def _metric_table(results: Dict[str, dict]) -> str:
    fields = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")
    lines = ["| Candidate | " + " | ".join(fields) + " |",
             "|---|" + "---:|" * len(fields)]
    for name, result in results.items():
        lines.append("| {0} | {1} |".format(
            name, " | ".join("{0:.4f}".format(result[key]) for key in fields)
        ))
    return "\n".join(lines)


def run_experiment(rows: Optional[Sequence[dict]] = None,
                   model_path: Path = MODEL_PATH,
                   report_path: Path = REPORT_PATH) -> dict:
    """Select on validation, refit train+validation, and evaluate held-out test."""
    if rows is None:
        rows = load_tickets()
    texts, labels, groups = prepare_department_rows(rows)
    splits = duplicate_aware_split(texts, labels, groups)
    class_names = sorted(set(labels))
    validation_results = {}
    fitted = {}
    for name, candidate in build_candidates().items():
        candidate.fit(_indexed(texts, splits["train"]), _indexed(labels, splits["train"]))
        validation_results[name] = evaluate(
            candidate, _indexed(texts, splits["validation"]),
            _indexed(labels, splits["validation"]), class_names
        )
        fitted[name] = candidate
    selected_name = max(
        validation_results,
        key=lambda name: (validation_results[name]["macro_f1"],
                          validation_results[name]["weighted_f1"],
                          validation_results[name]["accuracy"], name),
    )
    final_pipeline = build_candidates()[selected_name]
    final_indices = splits["train"] + splits["validation"]
    final_pipeline.fit(_indexed(texts, final_indices), _indexed(labels, final_indices))
    test_result = evaluate(
        final_pipeline, _indexed(texts, splits["test"]),
        _indexed(labels, splits["test"]), class_names
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, str(model_path))

    row_counts = {name: len(indices) for name, indices in splits.items()}
    class_counts = {
        name: dict(Counter(_indexed(labels, indices)))
        for name, indices in splits.items()
    }
    report = _build_report(row_counts, class_counts, validation_results,
                           selected_name, test_result, len(texts), len(set(groups)))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return {
        "selected_model": selected_name,
        "validation": validation_results,
        "test": test_result,
        "split_rows": row_counts,
        "split_class_counts": class_counts,
        "model_path": str(model_path),
        "report_path": str(report_path),
    }


def _build_report(row_counts: dict, class_counts: dict, validation: dict,
                  selected: str, test_result: dict, examples: int, groups: int) -> str:
    sections = [
        "# Department classifier experiment",
        "## Split methodology",
        "Used the {0}-fold `StratifiedGroupKFold` strategy with shuffle and seed `{1}`. "
        "One fold was held out for test, a second for validation, and the remaining three "
        "for training (approximately 60/20/20 by rows). Groups are case-folded, whitespace-normalized "
        "Body keys; each group is assigned to one split. The one blank Body was excluded. "
        "The only input feature is Body text. Tags, index, Priority, and Department are not features. "
        "TF-IDF is fit within each candidate pipeline on the training partition only.".format(
            SPLIT_FOLDS, RANDOM_SEED),
        "There are {0} usable rows and {1} distinct normalized body groups. Grouping matters because "
        "row-level random splitting could put a repeated ticket body in both training and evaluation, "
        "inflating results through memorization.".format(examples, groups),
        "### Split sizes and class counts",
        "| Split | Rows | Department counts |\n|---|---:|---|" + "\n" + "\n".join(
            "| {0} | {1} | `{2}` |".format(name, row_counts[name],
               json.dumps(class_counts[name], ensure_ascii=False, sort_keys=True))
            for name in ("train", "validation", "test")
        ),
        "## Validation model comparison",
        "Candidate selection used validation macro F1 as the primary criterion, with weighted F1 and accuracy as deterministic tie-breakers. Metrics are calculated on the actual held-out validation rows.",
        _metric_table(validation),
        "### Validation confusion matrices (rows = actual, columns = predicted)",
        "\n\n".join("#### {0}\n\n{1}".format(name, _confusion_table(value["confusion_matrix"], value["class_names"]))
                     for name, value in validation.items()),
        "## Selected model and untouched test result",
        "**Selected on validation:** {0}. The selected pipeline was refit on train plus validation; "
        "the test partition was not used for selection. The stored artifact is `models/department_classifier.joblib`.".format(selected),
        _metric_table({selected: test_result}),
        "### Test confusion matrix (rows = actual, columns = predicted)",
        _confusion_table(test_result["confusion_matrix"], test_result["class_names"]),
        "## Limitations",
        "The dataset contains repeated generic bodies with conflicting Department labels. Grouping prevents overlap but cannot remove label ambiguity. Group-aware stratification approximates natural class proportions; groups of different sizes make exact 60/20/20 row ratios and exact class proportions impossible simultaneously. Metrics describe this dataset split and do not establish production performance. The held-out test is a single sample of the source data.",
    ]
    return "\n\n".join(sections) + "\n"


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2, ensure_ascii=False))
