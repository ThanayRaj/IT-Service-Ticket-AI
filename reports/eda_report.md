# Dataset inspection and exploratory analysis

## Scope and source

Inspected `data/IT Support Ticket Data.csv` in its supplied form. The source file was not modified. CSV records were read with Python's standard CSV reader; this avoided assuming a schema from the brief or relying on the existing Python 3.7 environment's unavailable pandas package.

## Structure and quality

- **Rows:** 29,651
- **Columns:** 5: an empty-header column, `Body`, `Department`, `Priority`, and `Tags`.
- The empty-header field contains 29,651 unique values from 0 through 29,650. It is an exported row index, not a business feature, and must be excluded from modeling.
- All parsed fields are strings. `Body` has one blank value; the other fields have no blank values.
- There are **zero fully duplicated rows**.
- There are 25,056 distinct raw `Body` values, including the blank value. There are **4,595 repeated-body extra rows** (one is the single blank value); after trimming whitespace there are 25,054 distinct non-empty body strings.
- Repeated text is not always consistently labeled: 9 repeated non-empty body groups contain more than one Department label, and 7 contain more than one Priority label. These conflicting groups account for 10 Department rows and 8 Priority rows beyond each group's most frequent label. Examples are short generic texts such as “Seeking assistance” and “Assistance Required.”

## Text statistics

Statistics below use trimmed body text and include the one empty body in the distribution. Character percentiles use linear interpolation.

| Measure | Characters | Word tokens* |
|---|---:|---:|
| Minimum | 0 | 0 |
| 25th percentile | 208 | — |
| Median | 372 | 55 |
| Mean | — | 58.3 |
| 75th percentile | 538 | — |
| 90th percentile | 607 | — |
| 95th percentile | 705 | — |
| 99th percentile | 1,149.5 | — |
| Maximum | 2,422 | 366 |

\* Word counts are regex-based (`\w+`) token counts, not linguistic tokenization.

## Target distributions

### Department

| Department | Rows |
|---|---:|
| Technical Support | 8,617 |
| Product Support | 5,539 |
| Customer Service | 4,482 |
| IT Support | 3,500 |
| Billing and Payments | 3,017 |
| Returns and Exchanges | 1,467 |
| Service Outages and Maintenance | 1,157 |
| Sales and Pre-Sales | 885 |
| Human Resources | 568 |
| General Inquiry | 419 |

### Priority

| Priority | Rows |
|---|---:|
| medium | 12,126 |
| high | 11,512 |
| low | 6,013 |

Both labels are present with multiple classes and enough records to support supervised classification experiments. Department is moderately imbalanced; macro and weighted metrics should both be reported. Stratification is feasible in a holdout split, subject to duplicate-aware splitting.

## Department and Priority relationship

Counts by Department and Priority:

| Department | high | low | medium |
|---|---:|---:|---:|
| Billing and Payments | 886 | 632 | 1,499 |
| Customer Service | 828 | 1,464 | 2,190 |
| General Inquiry | 66 | 246 | 107 |
| Human Resources | 59 | 271 | 238 |
| IT Support | 1,657 | 347 | 1,496 |
| Product Support | 1,682 | 1,048 | 2,809 |
| Returns and Exchanges | 328 | 546 | 593 |
| Sales and Pre-Sales | 154 | 323 | 408 |
| Service Outages and Maintenance | 818 | 143 | 196 |
| Technical Support | 5,034 | 993 | 2,590 |

The target mix differs by Department. This association is expected to matter for interpretation, but the two target models should remain separate as specified.

## Tags and leakage review

`Tags` contains 12,946 distinct serialized tag-list patterns. Examples include combinations such as `['Bug', 'Performance', 'IT', 'Tech Support']`. Tags are potentially label-informative: some include department-like terms, and therefore must not be used as input for the Body-only classifiers. The index is likewise excluded.

Repeated bodies create a direct evaluation leakage risk if a random row split places the same body in both train and test. A few short repeated generic bodies also have inconsistent labels, demonstrating genuine ambiguity/noise rather than a reliable one-text/one-label mapping. Training/evaluation should therefore group identical normalized non-empty bodies into a single partition, while removing blank bodies from model fitting. Conflicting repeated labels should be reported and retained or resolved using a documented rule; they must not be silently relabeled. Any final test should remain untouched during candidate selection, with model choice made using training-only cross-validation or a separate validation partition.

No timestamp, resolution duration, or historical SLA outcome column exists. The data cannot support historical SLA breach prediction or time-to-resolution modeling. Application-created records may have real creation timestamps and can use configured thresholds for a transparent, explicitly assumption-based risk indicator.

## Modeling recommendation and decision

Proceed with separate Department and Priority text classifiers using `Body` only. Start with a reproducible scikit-learn pipeline that combines conservative text cleanup, word/character TF-IDF where appropriate, and interpretable classical candidates (Logistic Regression, Linear SVM, and Multinomial Naive Bayes). Keep vectorization inside each fitted pipeline so it is learned from training data only. Compare candidates using duplicate-group-aware validation and report accuracy, macro/weighted precision, recall, F1, and confusion matrices. Probability/confidence should only be surfaced if the selected estimator provides calibrated or otherwise defensible probabilities; a raw margin is not a probability.

The dataset supports the proposed classification architecture with the noted precautions. The next milestone can implement reusable loading/preprocessing and a reproducible EDA path. Model evaluation still needs a modern isolated Python environment; the available `python` command is Python 3.7.0 and does not have pandas installed. No packages have been installed.
