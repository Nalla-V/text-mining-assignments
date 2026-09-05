# Text Mining Assignments

Two assignments from the Text Mining course at LIACS, Leiden University (2025). Both build on
published tutorials, extend them, and report a proper evaluation. Each folder has the code and
the write-up.

The larger project from the same course — multi-label classification of Dutch election
manifestos — is in a separate repository.

## A1 — Text categorization

Multi-class classification on the full 20 Newsgroups benchmark, extending the scikit-learn
sparse-features tutorial from four categories to all twenty. Compares three classifiers across
three feature representations, then tunes the winning pair.

Evaluation is macro-averaged precision, recall and F1 across the twenty classes.

**Classifiers on TF–IDF:** Logistic Regression leads at F1 0.842, ahead of Complement Naive
Bayes (0.819) and k-Nearest Neighbours (0.753). kNN's weakness is the expected one for a
distance-based method in a high-dimensional sparse space.

**Representations:** TF–IDF beats raw counts and plain term frequency for every classifier.
The gap is widest for kNN, whose F1 more than doubles from 0.313 on raw counts to 0.753 on
TF–IDF — unnormalised counts make document length dominate the distance metric.

**Parameter sweep on Logistic Regression + TF–IDF:** character n-grams (3–5) give the best
result at F1 0.850, plausibly because they absorb the spelling variation and informal
word-forms common in newsgroup text. Word bigrams help slightly (0.846). Bigrams *alone*
collapse to 0.713 — the unigram signal is doing most of the work. Lowercasing and stopword
removal make almost no difference, and capping the vocabulary hurts.

Files: [`A1.ipynb`](A1-text-categorization/A1.ipynb),
[`report.pdf`](A1-text-categorization/report.pdf).
Data is fetched programmatically by scikit-learn — nothing to download.

## A2 — Sequence labelling (NER)

Named entity recognition on archaeological reports, fine-tuning `bert-base-cased` for token
classification over six entity types: artefact, construction, location, material, period and
species.

The interesting part is that a fine, headline-looking number hides two distinct failure modes.
Micro entity-F1 is 0.6607; Macro is 0.5494. The 0.11 gap is the whole story.

**Frequency drives type performance.** PER (1,213 training entities) reaches entity-F1 0.8123
and LOC 0.7604. MAT (150 instances) reaches 0.4211 and SPE (121 instances) 0.2500. The test
set contains just 2 SPE entities, so that number is close to meaningless — worth saying out
loud rather than reporting it as a result.

**Boundary detection fails separately from type detection.** B-MAT reaches F1 0.4842 while
I-MAT is 0.0000: the model finds where material entities start and never where they continue,
which breaks the span and halves full-entity performance. ART shows the same pattern despite
1,000 training examples (I-ART 0.4098), because artefacts are long descriptive phrases built
from common words — "Neolithic flint scraper" — where one wrong I-label invalidates the whole
entity.

So the two problems are different: rare types lack data, and multi-token types lack boundary
precision. Class-balanced sampling would address the first, not the second.

Files: [`ner.py`](A2-sequence-labelling/ner.py) (data conversion, fine-tuning, evaluation),
[`stats.py`](A2-sequence-labelling/stats.py) (corpus statistics),
[`report.pdf`](A2-sequence-labelling/report.pdf).

Data: the [ArchaeoNER English corpus](https://github.com/alexbrandsen/Archaeo-NER-data-English),
fold 1. Download `train.txt`, `val.txt` and `test.txt` into the `A2-sequence-labelling/` folder
before running.

```bash
python stats.py   # corpus statistics
python ner.py     # fine-tune and evaluate
```

## Context

Coursework for Text Mining, LIACS, Leiden University, 2025. Joint work with Luis Chial Sanchez
throughout; the work split for each assignment is stated in the corresponding report.
