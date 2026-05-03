from sklearn.metrics import f1_score, classification_report, confusion_matrix


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("F1 score:", f1_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return y_pred