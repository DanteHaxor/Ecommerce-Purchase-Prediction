from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train(X, y, preprocessor):
    X_train, X_test, y_train, y_test = split_data(X, y)

    dt = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=30,
        class_weight="balanced",
        random_state=42,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", dt),
        ]
    )

    pipe.fit(X_train, y_train)

    param_grid = {
        "model__max_depth": [4, 6, 8],
        "model__min_samples_leaf": [20, 30, 50],
    }

    grid = GridSearchCV(
        pipe,
        param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    return pipe, grid, X_train, X_test, y_train, y_test
