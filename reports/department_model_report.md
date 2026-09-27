# Department classifier experiment

## Split methodology

Used the 5-fold `StratifiedGroupKFold` strategy with shuffle and seed `42`. One fold was held out for test, a second for validation, and the remaining three for training (approximately 60/20/20 by rows). Groups are case-folded, whitespace-normalized Body keys; each group is assigned to one split. The one blank Body was excluded. The only input feature is Body text. Tags, index, Priority, and Department are not features. TF-IDF is fit within each candidate pipeline on the training partition only.

There are 29650 usable rows and 25044 distinct normalized body groups. Grouping matters because row-level random splitting could put a repeated ticket body in both training and evaluation, inflating results through memorization.

### Split sizes and class counts

| Split | Rows | Department counts |
|---|---:|---|
| train | 17790 | `{"Billing and Payments": 1811, "Customer Service": 2689, "General Inquiry": 251, "Human Resources": 341, "IT Support": 2100, "Product Support": 3322, "Returns and Exchanges": 881, "Sales and Pre-Sales": 531, "Service Outages and Maintenance": 694, "Technical Support": 5170}` |
| validation | 5930 | `{"Billing and Payments": 603, "Customer Service": 897, "General Inquiry": 84, "Human Resources": 113, "IT Support": 700, "Product Support": 1108, "Returns and Exchanges": 293, "Sales and Pre-Sales": 177, "Service Outages and Maintenance": 232, "Technical Support": 1723}` |
| test | 5930 | `{"Billing and Payments": 603, "Customer Service": 896, "General Inquiry": 84, "Human Resources": 114, "IT Support": 700, "Product Support": 1108, "Returns and Exchanges": 293, "Sales and Pre-Sales": 177, "Service Outages and Maintenance": 231, "Technical Support": 1724}` |

## Validation model comparison

Candidate selection used validation macro F1 as the primary criterion, with weighted F1 and accuracy as deterministic tie-breakers. Metrics are calculated on the actual held-out validation rows.

All candidates used the same fitted pipeline: whitespace/HTML-entity cleanup, word TF-IDF (unigrams and bigrams, `min_df=2`, `max_df=0.98`, sublinear term frequency, Unicode accent stripping, at most 200,000 features), followed by the candidate estimator. Runtime: Python 3.12.14 and scikit-learn 1.9.1.

| Candidate | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5358 | 0.7165 | 0.3859 | 0.4385 | 0.5188 |
| Linear SVM | 0.6039 | 0.7057 | 0.5298 | 0.5876 | 0.6010 |
| Multinomial Naive Bayes | 0.4086 | 0.5011 | 0.2113 | 0.2048 | 0.3479 |

### Validation confusion matrices (rows = actual, columns = predicted)

#### Logistic Regression

| Actual \ Predicted | Billing and Payments | Customer Service | General Inquiry | Human Resources | IT Support | Product Support | Returns and Exchanges | Sales and Pre-Sales | Service Outages and Maintenance | Technical Support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Billing and Payments | 430 | 57 | 0 | 0 | 15 | 21 | 1 | 0 | 0 | 79 |
| Customer Service | 11 | 439 | 0 | 0 | 31 | 135 | 3 | 2 | 1 | 275 |
| General Inquiry | 0 | 15 | 8 | 0 | 6 | 24 | 0 | 0 | 0 | 31 |
| Human Resources | 5 | 11 | 0 | 19 | 7 | 24 | 0 | 0 | 0 | 47 |
| IT Support | 1 | 61 | 0 | 0 | 229 | 74 | 1 | 0 | 4 | 330 |
| Product Support | 17 | 109 | 0 | 0 | 23 | 517 | 5 | 4 | 1 | 432 |
| Returns and Exchanges | 10 | 51 | 0 | 0 | 8 | 59 | 62 | 3 | 0 | 100 |
| Sales and Pre-Sales | 6 | 60 | 0 | 0 | 6 | 36 | 1 | 25 | 0 | 43 |
| Service Outages and Maintenance | 5 | 6 | 0 | 0 | 21 | 21 | 0 | 0 | 109 | 70 |
| Technical Support | 9 | 126 | 0 | 0 | 45 | 180 | 6 | 5 | 13 | 1339 |

#### Linear SVM

| Actual \ Predicted | Billing and Payments | Customer Service | General Inquiry | Human Resources | IT Support | Product Support | Returns and Exchanges | Sales and Pre-Sales | Service Outages and Maintenance | Technical Support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Billing and Payments | 475 | 34 | 0 | 2 | 16 | 21 | 2 | 3 | 0 | 50 |
| Customer Service | 13 | 502 | 1 | 0 | 38 | 110 | 7 | 6 | 2 | 218 |
| General Inquiry | 0 | 11 | 32 | 0 | 2 | 16 | 1 | 0 | 0 | 22 |
| Human Resources | 5 | 10 | 1 | 43 | 6 | 16 | 0 | 0 | 1 | 31 |
| IT Support | 4 | 57 | 0 | 0 | 342 | 58 | 5 | 0 | 6 | 228 |
| Product Support | 24 | 103 | 0 | 2 | 45 | 585 | 13 | 8 | 2 | 326 |
| Returns and Exchanges | 5 | 43 | 0 | 0 | 9 | 43 | 128 | 3 | 0 | 62 |
| Sales and Pre-Sales | 5 | 45 | 0 | 0 | 6 | 25 | 4 | 67 | 2 | 23 |
| Service Outages and Maintenance | 4 | 5 | 0 | 2 | 18 | 10 | 0 | 5 | 145 | 43 |
| Technical Support | 23 | 117 | 0 | 5 | 78 | 194 | 19 | 7 | 18 | 1262 |

#### Multinomial Naive Bayes

| Actual \ Predicted | Billing and Payments | Customer Service | General Inquiry | Human Resources | IT Support | Product Support | Returns and Exchanges | Sales and Pre-Sales | Service Outages and Maintenance | Technical Support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Billing and Payments | 313 | 142 | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 139 |
| Customer Service | 5 | 397 | 0 | 0 | 0 | 97 | 0 | 0 | 0 | 398 |
| General Inquiry | 0 | 17 | 0 | 0 | 0 | 22 | 0 | 0 | 0 | 45 |
| Human Resources | 1 | 32 | 0 | 0 | 0 | 12 | 0 | 0 | 0 | 68 |
| IT Support | 0 | 128 | 0 | 0 | 37 | 37 | 0 | 0 | 0 | 498 |
| Product Support | 1 | 195 | 0 | 0 | 1 | 268 | 0 | 0 | 0 | 643 |
| Returns and Exchanges | 7 | 73 | 0 | 0 | 0 | 69 | 6 | 0 | 0 | 138 |
| Sales and Pre-Sales | 7 | 75 | 0 | 0 | 0 | 29 | 0 | 0 | 0 | 66 |
| Service Outages and Maintenance | 1 | 22 | 0 | 0 | 0 | 19 | 0 | 0 | 6 | 184 |
| Technical Support | 1 | 244 | 0 | 0 | 0 | 82 | 0 | 0 | 0 | 1396 |

## Selected model and untouched test result

**Selected on validation:** Linear SVM. The selected pipeline was refit on train plus validation; the test partition was not used for selection. The stored artifact is `models/department_classifier.joblib`.

| Candidate | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 |
|---|---:|---:|---:|---:|---:|
| Linear SVM | 0.6528 | 0.7529 | 0.5981 | 0.6555 | 0.6502 |

### Test confusion matrix (rows = actual, columns = predicted)

| Actual \ Predicted | Billing and Payments | Customer Service | General Inquiry | Human Resources | IT Support | Product Support | Returns and Exchanges | Sales and Pre-Sales | Service Outages and Maintenance | Technical Support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Billing and Payments | 500 | 30 | 2 | 0 | 11 | 30 | 3 | 2 | 0 | 25 |
| Customer Service | 25 | 509 | 0 | 0 | 39 | 98 | 13 | 6 | 2 | 204 |
| General Inquiry | 1 | 18 | 42 | 0 | 4 | 3 | 0 | 0 | 1 | 15 |
| Human Resources | 5 | 12 | 0 | 67 | 1 | 10 | 1 | 0 | 1 | 17 |
| IT Support | 8 | 44 | 0 | 2 | 337 | 50 | 4 | 5 | 8 | 242 |
| Product Support | 18 | 86 | 0 | 0 | 54 | 678 | 8 | 7 | 5 | 252 |
| Returns and Exchanges | 3 | 46 | 0 | 0 | 6 | 39 | 143 | 2 | 1 | 53 |
| Sales and Pre-Sales | 1 | 25 | 0 | 0 | 5 | 33 | 3 | 85 | 0 | 25 |
| Service Outages and Maintenance | 1 | 12 | 0 | 1 | 8 | 11 | 1 | 0 | 149 | 48 |
| Technical Support | 9 | 131 | 2 | 2 | 62 | 136 | 6 | 7 | 8 | 1361 |

## Limitations

The dataset contains repeated generic bodies with conflicting Department labels. Grouping prevents overlap but cannot remove label ambiguity. Group-aware stratification approximates natural class proportions; groups of different sizes make exact 60/20/20 row ratios and exact class proportions impossible simultaneously. Metrics describe this dataset split and do not establish production performance. The held-out test is a single sample of the source data.
