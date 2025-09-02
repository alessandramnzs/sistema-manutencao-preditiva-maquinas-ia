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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "bootcamp_train_clean.csv")
METRICS_PATH = os.path.join(BASE_DIR, "visualizacoes", "metricas_multiclasse.json")
MODEL_PATH = os.path.join(BASE_DIR, "api", "model_multiclasse.pkl")


def build_multiclass_target(row: pd.Series) -> int:
    if row["FDF (Falha Desgaste Ferramenta)"] == 1:
        return 1
    if row["FDC (Falha Dissipacao Calor)"] == 1:
        return 2
    if row["FP (Falha Potencia)"] == 1:
        return 3
    if row["FTE (Falha Tensao Excessiva)"] == 1:
        return 4
    if row["FA (Falha Aleatoria)"] == 1:
        return 5
    return 0


def main():
    df = pd.read_csv(DATA_PATH)

    required_cols = [
        "FDF (Falha Desgaste Ferramenta)",
        "FDC (Falha Dissipacao Calor)",
        "FP (Falha Potencia)",
        "FTE (Falha Tensao Excessiva)",
        "FA (Falha Aleatoria)",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {missing}")

    df["classe_falha"] = df.apply(build_multiclass_target, axis=1).astype(int)

    target = df["classe_falha"]
    features = df.drop(
        columns=[
            "falha_maquina",
            "id",
            "id_produto",
            "FDF (Falha Desgaste Ferramenta)",
            "FDC (Falha Dissipacao Calor)",
            "FP (Falha Potencia)",
            "FTE (Falha Tensao Excessiva)",
            "FA (Falha Aleatoria)",
            "classe_falha",
        ],
        errors="ignore",
    )

    class_counts = target.value_counts().sort_index()
    rare_classes = class_counts[class_counts < 2]

    if not rare_classes.empty:
        print(
            f"[AVISO] Classes raras detectadas (menos de 2 amostras): "
            f"{rare_classes.to_dict()}. Usando split sem estratificação."
        )
        stratify_arg = None
    else:
        stratify_arg = target

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=stratify_arg,
        random_state=42,
        shuffle=True,
    )

    categorical = ["tipo"] if "tipo" in features.columns else []
    preprocessor = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough",
    )

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", multi_class="multinomial"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=250, class_weight="balanced", random_state=42
        ),
        "XGBoost": XGBClassifier(
            eval_metric="mlogloss",
            n_estimators=350,
            max_depth=6,
        ),
    }

    model_results = {}
    best_pipeline = None
    best_model_name = None
    best_f1_macro = -1.0

    for name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        confusion = confusion_matrix(y_test, predictions).tolist()

        model_results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "f1_macro": float(report["macro avg"]["f1-score"]),
            "classification_report": report,
            "confusion_matrix": confusion,
        }

        if model_results[name]["f1_macro"] > best_f1_macro:
            best_f1_macro = model_results[name]["f1_macro"]
            best_model_name = name
            best_pipeline = pipeline

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    summary = {
        "best_model": best_model_name,
        "best_f1_macro": best_f1_macro,
        "all_results": model_results,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    if best_pipeline is not None:
        joblib.dump(best_pipeline, MODEL_PATH)
        print(f"O melhor modelo é o {best_model_name} com F1 Macro={best_f1_macro:.3f} e foi salvo em {MODEL_PATH}")


if __name__ == "__main__":
    main()
