import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def load_and_prepare_data(path):
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "text"]
    )

    df["clean_text"] = df["text"].apply(clean_text)

    X = df["clean_text"]
    y = df["label"].map({"ham": 0, "spam": 1})

    return X, y
def vectorize_and_split(X, y):
    vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words="english"
    )

    X_vectors = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vectors,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X, y = load_and_prepare_data("C:/Users/LISHA/Documents/prjct 4 yr/nlp-ml-pipeline/data/raw/sms_spam.tsv")
    X_train, X_test, y_train, y_test = vectorize_and_split(X, y)

    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)
