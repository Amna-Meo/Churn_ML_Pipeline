import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")


def train_models(X_train, y_train, preprocessor):
    results = {}

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            random_state=42, n_jobs=-1, class_weight="balanced"
        ),
    }

    param_grids = {
        "LogisticRegression": {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__penalty": ["l1", "l2"],
            "classifier__solver": ["saga"],
        },
        "RandomForest": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [5, 10, 15],
            "classifier__min_samples_split": [2, 5, 10],
        },
    }

    best_models = {}
    best_scores = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])

        grid_search = GridSearchCV(
            pipeline, param_grids[name], cv=5, scoring="f1", n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)

        best_models[name] = grid_search.best_estimator_
        best_scores[name] = grid_search.best_score_

        print(f"Best {name} F1 Score: {grid_search.best_score_:.4f}")
        print(f"Best params: {grid_search.best_params_}")

        results[name] = {
            "best_model": grid_search.best_estimator_,
            "best_score": grid_search.best_score_,
            "best_params": grid_search.best_params_,
        }

    best_model_name = max(best_scores, key=best_scores.get)
    best_model = best_models[best_model_name]

    print(f"\n{'=' * 50}")
    print(f"Best Overall Model: {best_model_name}")
    print(f"Best F1 Score: {best_scores[best_model_name]:.4f}")

    return best_model, best_model_name, results


def export_pipeline(pipeline, filepath):
    joblib.dump(pipeline, filepath)
    print(f"\nPipeline exported to: {filepath}")
