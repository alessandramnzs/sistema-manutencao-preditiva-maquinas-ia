import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    accuracy_score,
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "bootcamp_train_clean.csv")
METRICS_PATH = os.path.join(BASE_DIR, "visualizacoes", "metricas_multirrotulo.json")
MODEL_PATH = os.path.join(BASE_DIR, "api", "model_multirrotulo.pkl")


def main():
    df = pd.read_csv(DATA_PATH)

    target_columns = [
        "FDF (Falha Desgaste Ferramenta)",
        "FDC (Falha Dissipacao Calor)",
        "FP (Falha Potencia)",
        "FTE (Falha Tensao Excessiva)",
        "FA (Falha Aleatoria)",
    ]

    features = df.drop(columns=["falha_maquina", "id", "id_produto"] + target_columns, errors="ignore")
    targets = df[target_columns].applymap(lambda x: 1 if str(x).strip().lower() in {"1", "true", "yes"} else 0)

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    categorical = ["tipo"] if "tipo" in features.columns else []
    preprocessor = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough",
    )

    models = {
        "RandomForest": MultiOutputClassifier(
            RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
        ),
        "XGBoost": MultiOutputClassifier(
            XGBClassifier(eval_metric="logloss", n_estimators=300, max_depth=6)
        ),
        "MLP": MultiOutputClassifier(
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42)
        ),
    }

    model_results = {}
    best_pipeline = None
    best_model_name = None
    best_f1_micro = -1.0

    for name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        metrics = {
            "f1_micro": float(f1_score(y_test, predictions, average="micro")),
            "f1_macro": float(f1_score(y_test, predictions, average="macro")),
            "precision_micro": float(precision_score(y_test, predictions, average="micro")),
            "recall_micro": float(recall_score(y_test, predictions, average="micro")),
            "hamming_loss": float(hamming_loss(y_test, predictions)),
            "subset_accuracy": float(accuracy_score(y_test, predictions)),
        }
        model_results[name] = metrics

        if metrics["f1_micro"] > best_f1_micro:
            best_f1_micro = metrics["f1_micro"]
            best_model_name = name
            best_pipeline = pipeline

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    summary = {
        "best_model": best_model_name,
        "best_f1_micro": best_f1_micro,
        "all_results": model_results,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    if best_pipeline is not None:
        joblib.dump(best_pipeline, MODEL_PATH)
        print(f"O melhor modelo é o {best_model_name} com F1 micro={best_f1_micro:.3f} e foi salvo em {MODEL_PATH}")


if __name__ == "__main__":
    main()
