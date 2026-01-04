from sklearn.metrics import classification_report, confusion_matrix
import joblib

from preprocess import load_and_prepare_data, vectorize_and_split


def evaluate_model():
    # Load data
    X, y = load_and_prepare_data("data/raw/sms_spam.tsv")
    X_train, X_test, y_train, y_test = vectorize_and_split(X, y)

    # Load trained model
    model = joblib.load("spam_classifier.pkl")

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    evaluate_model()