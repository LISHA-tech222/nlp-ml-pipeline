import joblib
from sklearn.linear_model import LogisticRegression

from preprocess import load_and_prepare_data, vectorize_and_split


def train_model():
    X, y = load_and_prepare_data("data/raw/sms_spam.tsv")
    X_train, X_test, y_train, y_test = vectorize_and_split(X, y)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


if __name__ == "__main__":
    model, X_test, y_test = train_model()

    joblib.dump(model, "spam_classifier.pkl")

    print("Model trained and saved as spam_classifier.pkl")