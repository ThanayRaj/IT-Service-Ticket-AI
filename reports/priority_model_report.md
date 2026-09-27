# Priority classifier experiment

## Split methodology

Reused the exact Milestone 3 train/validation/test assignments: the same ordered non-empty Body rows, normalized Body groups, five-fold `StratifiedGroupKFold`, shuffle, and seed `42`. The original Department-stratified assignment was reproduced solely to retain those partitions; Department was not supplied to any classifier, vectorizer, or prediction pipeline. The split remains approximately 60/20/20 by rows. Blank Body rows were excluded. All candidate features are Body text only, and TF-IDF is fitted within each candidate pipeline on the training partition.

The data contain 29650 usable rows and 25044 normalized Body groups. Overall Priority counts are `{"high": 11512, "low": 6013, "medium": 12126}`. Keeping each group intact avoids the same repeated text appearing in train and evaluation. Priority label counts are shown for each reused split.

### Split sizes and Priority counts

| Split | Rows | Priority counts |
|---|---:|---|
| train | 17790 | `{"high": 6919, "low": 3617, "medium": 7254}` |
| validation | 5930 | `{"high": 2298, "low": 1206, "medium": 2426}` |
| test | 5930 | `{"high": 2294, "low": 1190, "medium": 2446}` |

## Validation candidate comparison

Candidates were ranked by validation macro F1, then weighted F1, then accuracy as deterministic tie-breakers. Metrics below are measured on the validation split.

| Candidate | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6169 | 0.6177 | 0.5691 | 0.5768 | 0.6061 |
| Linear SVM | 0.6470 | 0.6405 | 0.6140 | 0.6221 | 0.6426 |
| Multinomial Naive Bayes | 0.5536 | 0.6534 | 0.4730 | 0.4443 | 0.5077 |

### Validation confusion matrices (rows = actual, columns = predicted)

#### Logistic Regression

| Actual \ Predicted | high | low | medium |
|---|---:|---:|---:|
| high | 1585 | 92 | 621 |
| low | 289 | 391 | 526 |
| medium | 592 | 152 | 1682 |

#### Linear SVM

| Actual \ Predicted | high | low | medium |
|---|---:|---:|---:|
| high | 1621 | 140 | 537 |
| low | 252 | 535 | 419 |
| medium | 550 | 195 | 1681 |

#### Multinomial Naive Bayes

| Actual \ Predicted | high | low | medium |
|---|---:|---:|---:|
| high | 1470 | 11 | 817 |
| low | 383 | 77 | 746 |
| medium | 688 | 2 | 1736 |

## Selected model and untouched test result

**Selected on validation:** Linear SVM. It was refit on train plus validation; the held-out test partition was not used for selection. The saved artifact is `models/priority_classifier.joblib`.

| Candidate | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 |
|---|---:|---:|---:|---:|---:|
| Linear SVM | 0.6860 | 0.6864 | 0.6622 | 0.6711 | 0.6840 |

### Test confusion matrix (rows = actual, columns = predicted)

| Actual \ Predicted | high | low | medium |
|---|---:|---:|---:|
| high | 1678 | 102 | 514 |
| low | 206 | 644 | 340 |
| medium | 507 | 193 | 1746 |

## Repeated-body label ambiguity

Among normalized non-empty Body groups, 9 groups have conflicting Priority labels, with 10 rows beyond the within-group majority label. Examples of the normalized generic bodies include: `requesting assistance`; `assistance needed`; `seeking assistance`; `assistance required`; `offer insights into digital strategies`. These are retained as supplied; group splitting prevents overlap but cannot resolve label ambiguity.

## Limitations

Priority is imbalanced, so macro F1 is the primary selection measure; class-weighting was not added, and the minority class can still be missed. Reusing Department-stratified partitions preserves comparability with Milestone 3 rather than stratifying this split anew on Priority; observed Priority proportions remain close across the three partitions. Metrics describe one held-out sample from this dataset and do not establish production performance.
