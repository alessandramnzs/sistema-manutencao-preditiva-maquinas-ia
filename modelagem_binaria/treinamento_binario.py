import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "bootcamp_train_clean.csv")
METRICS_PATH = os.path.join(BASE_DIR, "visualizacoes", "metricas_binario.json")
MODEL_PATH = os.path.join(BASE_DIR, "api", "model_binario.pkl")


def main():
    df = pd.read_csv(DATA_PATH)

    target = df["FTE (Falha Tensao Excessiva)"]
    features = df.drop(
        columns=[
            "falha_maquina",
            "FDF (Falha Desgaste Ferramenta)",
            "FDC (Falha Dissipacao Calor)",
            "FP (Falha Potencia)",
            "FTE (Falha Tensao Excessiva)",
            "FA (Falha Aleatoria)",
            "id",
            "id_produto",
        ],
        errors="ignore",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=42
    )

    categorical = ["tipo"] if "tipo" in features.columns else []
    preprocessor = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough",
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            n_estimators=400,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
        ),
    }

    model_results = {}
    best_pipeline = None
    best_model_name = None
    best_f1 = -1.0

    for name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        try:
            probabilities = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            probabilities = None

        report = classification_report(y_test, predictions, output_dict=True)
        confusion = confusion_matrix(y_test, predictions).tolist()
        roc_auc = roc_auc_score(y_test, probabilities) if probabilities is not None else None

        model_results[name] = {
            "classification_report": report,
            "confusion_matrix": confusion,
            "roc_auc": roc_auc,
        }

        f1_positive = report.get("1", {}).get("f1-score", 0.0)
        if f1_positive > best_f1:
            best_f1 = f1_positive
            best_model_name = name
            best_pipeline = pipeline

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    summary = {
        "best_model": best_model_name,
        "best_f1_score": best_f1,
        "all_results": model_results,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    if best_pipeline is not None:
        joblib.dump(best_pipeline, MODEL_PATH)
        print(f"O melhor modelo é o {best_model_name} com F1={best_f1:.3f} e foi salvo em {MODEL_PATH}")


if __name__ == "__main__":
    main()
