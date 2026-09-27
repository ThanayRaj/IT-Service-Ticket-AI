"""Priority model selection using the exact group split from Milestone 3."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence

import joblib

from src.data_loader import load_tickets
from src.preprocessing import normalized_body_key
from src.train_department import (
    RANDOM_SEED,
    _confusion_table,
    _indexed,
    _metric_table,
    build_candidates,
    duplicate_aware_split,
    evaluate,
    prepare_department_rows,
)


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "priority_classifier.joblib"
REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "priority_model_report.md"


def prepare_priority_rows(rows: Sequence[dict]):
    """Build Body-only model inputs and Priority targets, excluding blank Body."""
    texts, _, groups = prepare_department_rows(rows)
    labels = [str(row.get("Priority", "")).strip() for row in rows
              if str(row.get("Body", "")).strip()]
    if len(texts) != len(labels):
        raise ValueError("Body and Priority rows do not align after blank-body filtering")
    return texts, labels, groups


def _reuse_department_split(rows, texts, groups):
    """Reproduce the exact Milestone 3 assignments from the same ordered rows.

    Department labels are used only as the stratification labels that defined
    the prior split. They are never passed to a classifier or vectorizer.
    """
    department_texts, department_labels, department_groups = prepare_department_rows(rows)
    if texts != department_texts or groups != department_groups:
        raise ValueError("Priority rows do not match the Milestone 3 Body ordering")
    return duplicate_aware_split(department_texts, department_labels, department_groups,
                                 random_seed=RANDOM_SEED)


def _priority_conflicts(rows):
    grouped = defaultdict(Counter)
    for row in rows:
        body = row.get("Body", "")
        if normalized_body_key(body):
            grouped[normalized_body_key(body)][str(row.get("Priority", "")).strip()] += 1
    conflicting = [(key, counts) for key, counts in grouped.items() if len(counts) > 1]
    discordant = sum(sum(counts.values()) - max(counts.values())
                     for _, counts in conflicting)
    examples = [key[:80] for key, _ in conflicting[:5]]
    return len(conflicting), discordant, examples


def _priority_report(row_counts, class_counts, validation, selected,
                     test_result, examples, groups, conflict_data):
    conflict_groups, discordant_rows, conflict_examples = conflict_data
    overall_counts = Counter()
    for counts in class_counts.values():
        overall_counts.update(counts)
    sections = [
        "# Priority classifier experiment",
        "## Split methodology",
        "Reused the exact Milestone 3 train/validation/test assignments: the same ordered non-empty Body rows, normalized Body groups, five-fold `StratifiedGroupKFold`, shuffle, and seed `{0}`. The original Department-stratified assignment was reproduced solely to retain those partitions; Department was not supplied to any classifier, vectorizer, or prediction pipeline. The split remains approximately 60/20/20 by rows. Blank Body rows were excluded. All candidate features are Body text only, and TF-IDF is fitted within each candidate pipeline on the training partition.".format(RANDOM_SEED),
        "The data contain {0} usable rows and {1} normalized Body groups. Overall Priority counts are `{2}`. Keeping each group intact avoids the same repeated text appearing in train and evaluation. Priority label counts are shown for each reused split.".format(examples, groups, json.dumps(dict(overall_counts), sort_keys=True)),
        "### Split sizes and Priority counts",
        "| Split | Rows | Priority counts |\n|---|---:|---|\n" + "\n".join(
            "| {0} | {1} | `{2}` |".format(name, row_counts[name],
                json.dumps(class_counts[name], sort_keys=True))
            for name in ("train", "validation", "test")
        ),
        "## Validation candidate comparison",
        "Candidates were ranked by validation macro F1, then weighted F1, then accuracy as deterministic tie-breakers. Metrics below are measured on the validation split.",
        _metric_table(validation),
        "### Validation confusion matrices (rows = actual, columns = predicted)",
        "\n\n".join("#### {0}\n\n{1}".format(
            name, _confusion_table(result["confusion_matrix"], result["class_names"])
        ) for name, result in validation.items()),
        "## Selected model and untouched test result",
        "**Selected on validation:** {0}. It was refit on train plus validation; the held-out test partition was not used for selection. The saved artifact is `models/priority_classifier.joblib`.".format(selected),
        _metric_table({selected: test_result}),
        "### Test confusion matrix (rows = actual, columns = predicted)",
        _confusion_table(test_result["confusion_matrix"], test_result["class_names"]),
        "## Repeated-body label ambiguity",
        "Among normalized non-empty Body groups, {0} groups have conflicting Priority labels, with {1} rows beyond the within-group majority label. Examples of the normalized generic bodies include: `{2}`. These are retained as supplied; group splitting prevents overlap but cannot resolve label ambiguity.".format(
            conflict_groups, discordant_rows, "`; `".join(conflict_examples)),
        "## Limitations",
        "Priority is imbalanced, so macro F1 is the primary selection measure; class-weighting was not added, and the minority class can still be missed. Reusing Department-stratified partitions preserves comparability with Milestone 3 rather than stratifying this split anew on Priority; observed Priority proportions remain close across the three partitions. Metrics describe one held-out sample from this dataset and do not establish production performance.",
    ]
    return "\n\n".join(sections) + "\n"


def run_experiment(rows: Optional[Sequence[dict]] = None,
                   model_path: Path = MODEL_PATH,
                   report_path: Path = REPORT_PATH) -> dict:
    """Select on validation, refit the selected candidate, and test once."""
    if rows is None:
        rows = load_tickets()
    texts, labels, groups = prepare_priority_rows(rows)
    splits = _reuse_department_split(rows, texts, groups)
    class_names = sorted(set(labels))

    validation_results = {}
    for name, candidate in build_candidates().items():
        candidate.fit(_indexed(texts, splits["train"]), _indexed(labels, splits["train"]))
        validation_results[name] = evaluate(
            candidate, _indexed(texts, splits["validation"]),
            _indexed(labels, splits["validation"]), class_names
        )

    selected_name = max(
        validation_results,
        key=lambda name: (validation_results[name]["macro_f1"],
                          validation_results[name]["weighted_f1"],
                          validation_results[name]["accuracy"], name),
    )
    final_pipeline = build_candidates()[selected_name]
    final_indices = splits["train"] + splits["validation"]
    final_pipeline.fit(_indexed(texts, final_indices), _indexed(labels, final_indices))
    test_result = evaluate(final_pipeline, _indexed(texts, splits["test"]),
                           _indexed(labels, splits["test"]), class_names)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, str(model_path))

    row_counts = {name: len(indices) for name, indices in splits.items()}
    class_counts = {name: dict(Counter(_indexed(labels, indices)))
                    for name, indices in splits.items()}
    report = _priority_report(row_counts, class_counts, validation_results,
                              selected_name, test_result, len(texts), len(set(groups)),
                              _priority_conflicts(rows))
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


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2, ensure_ascii=False))
