# NLP ML Pipeline

End-to-end machine learning pipeline for SMS spam classification using TF-IDF features and Logistic Regression.

## Pipeline

1. **Preprocessing** ([src/preprocess.py](src/preprocess.py)): lowercases text and strips non-alphabetic characters.
2. **Feature extraction**: TF-IDF vectorizer (3000 features, English stop words removed).
3. **Split**: 80/20 stratified train/test split (`random_state=42`).
4. **Training** ([src/train.py](src/train.py)): Logistic Regression with `class_weight="balanced"` to handle class imbalance; saved to `spam_classifier.pkl`.
5. **Evaluation** ([src/evaluate.py](src/evaluate.py)): precision, recall, F1-score and confusion matrix.

## Project structure

```
data/raw/sms_spam.tsv    # labelled SMS messages (tab-separated: label, text)
data/processed/
notebooks/exploration.ipynb
src/preprocess.py
src/train.py
src/evaluate.py
requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Usage

Run from the project root:

```bash
python src/preprocess.py   # sanity check: prints train/test shapes
python src/train.py        # trains and saves spam_classifier.pkl
python src/evaluate.py     # prints classification report and confusion matrix
```

Note: `evaluate.py` rebuilds the TF-IDF features and split from the raw data, so it must be run after `train.py` with the same preprocessing settings.

## Results

On the imbalanced dataset, the model achieved:

- Spam precision: 0.91
- Spam recall: 0.91
- Spam F1-score: 0.91

Precision, recall, F1-score and the confusion matrix are used instead of accuracy because of the class imbalance.

## Tech stack

Python, scikit-learn, pandas, joblib
