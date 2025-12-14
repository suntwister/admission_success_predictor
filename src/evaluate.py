import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def evaluate_classification_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test,y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    return metrics

def plot_confusion_matrix(model, X_test, y_test, title="Confussion Matrix"):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.show()

def plot_feature_importance(model, feature_names, title="Feature Importance"):
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not support feature importance")
    
    importances = model.feature_importances_

    plt.figure(figsize=(8,5))
    plt.barh(feature_names, importances)
    plt.title(title)
    plt.tight_layout()
    plt.show()
