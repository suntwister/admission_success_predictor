from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=200, random_state=42):
    model = RandomForestClassifier(
        n_estimators= n_estimators,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model
