import logging
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


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
            "classifier__l1_ratio": [0, 0.5, 1],
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
        logger.info(f"Training {name}...")

        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])

        grid_search = GridSearchCV(
            pipeline, param_grids[name], cv=5, scoring="f1", n_jobs=-1, verbose=0
        )

        grid_search.fit(X_train, y_train)

        best_models[name] = grid_search.best_estimator_
        best_scores[name] = grid_search.best_score_

        logger.info(f"Best {name} F1 Score: {grid_search.best_score_:.4f}")
        logger.info(f"Best params: {grid_search.best_params_}")

        results[name] = {
            "best_model": grid_search.best_estimator_,
            "best_score": grid_search.best_score_,
            "best_params": grid_search.best_params_,
        }

    best_model_name = max(best_scores, key=best_scores.get)
    best_model = best_models[best_model_name]

    logger.info(f"Best Overall Model: {best_model_name}")
    logger.info(f"Best F1 Score: {best_scores[best_model_name]:.4f}")

    return best_model, best_model_name, results
