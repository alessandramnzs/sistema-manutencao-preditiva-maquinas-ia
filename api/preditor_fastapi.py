from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse, JSONResponse
import os
import joblib
import pandas as pd

app = FastAPI(title="Preditor - Bootcamp CDIA")

BASE_DIR = os.path.dirname(__file__)
ML_MODEL = os.path.join(BASE_DIR, "model_multirrotulo.pkl")
BIN_MODEL = os.path.join(BASE_DIR, "model_binario.pkl")
MC_MODEL = os.path.join(BASE_DIR, "model_multiclasse.pkl")

modelo_multi = joblib.load(ML_MODEL) if os.path.exists(ML_MODEL) else None
modelo_bin = joblib.load(BIN_MODEL) if os.path.exists(BIN_MODEL) else None
modelo_mc = joblib.load(MC_MODEL) if os.path.exists(MC_MODEL) else None


def _multilabel_probabilities(model, X: pd.DataFrame):
    try:
        probabilities = model.predict_proba(X)
    except Exception:
        X_t = model.named_steps["prep"].transform(X)
        estimators = model.named_steps["clf"].estimators_
        probabilities = [est.predict_proba(X_t) for est in estimators]

    cols = []
    for p in probabilities:
        if isinstance(p, (list, tuple)):
            p = p[0]
        cols.append(p[:, 1])
    return pd.DataFrame({i: col for i, col in enumerate(cols)})


@app.post("/predict/multilabel")
async def predict_multilabel(file: UploadFile = File(...)):
    if modelo_multi is None:
        return JSONResponse(status_code=400, content={"error": "Não foi possível encontrar o modelo multirrótulo"})

    df = pd.read_csv(file.file, sep=None, engine="python")
    if "id" not in df.columns:
        return JSONResponse(status_code=400, content={"error": "O dataset precisa ter a coluna 'id' "})

    ids = df["id"].astype(str)
    X = df.drop(columns=["id"], errors="ignore")
    probability_matrix = _multilabel_probabilities(modelo_multi, X)

    out = pd.DataFrame({"id": ids})
    mapping = {0: "falha_1", 1: "falha_2", 2: "falha_3", 3: "falha_4", 4: "falha_5", 5: "falha_6"}
    for i, name in mapping.items():
        out[name] = probability_matrix[i] if i in probability_matrix.columns else 0.0
    out["falha_outros"] = 0.0

    for c in list(mapping.values()) + ["falha_outros"]:
        out[c] = out[c].clip(0, 1).astype(float).round(6)

    ordered = ["id", "falha_1", "falha_2", "falha_3", "falha_4", "falha_5", "falha_6", "falha_outros"]
    out = out[ordered]

    out_csv = os.path.join(BASE_DIR, "predicoes_multilabel.csv")
    out.to_csv(out_csv, index=False)
    return FileResponse(out_csv, filename="predicoes_para_api.csv")


@app.post("/predict/binary_fte")
async def predict_binary(file: UploadFile = File(...), threshold: float = Query(0.5)):
    if modelo_bin is None:
        return JSONResponse(status_code=400, content={"error": "Não foi possível encontrar o modelo binário"})

    df = pd.read_csv(file.file, sep=None, engine="python")
    if "id" not in df.columns:
        return JSONResponse(status_code=400, content={"error": "O dataset precisa ter a coluna 'id' "})

    ids = df["id"].astype(str)
    X = df.drop(columns=["id"], errors="ignore")

    try:
        probabilities = modelo_bin.predict_proba(X)[:, 1]
        y_pred = (probabilities >= threshold).astype(int)
    except Exception:
        y_pred = modelo_bin.predict(X)
        probabilities = None

    out = pd.DataFrame({"id": ids, "FTE_pred": y_pred})
    if probabilities is not None:
        out["FTE_score"] = probabilities

    out_csv = os.path.join(BASE_DIR, "predicoes_binario.csv")
    out.to_csv(out_csv, index=False)
    return FileResponse(out_csv, filename="predicoes_binario.csv")


@app.post("/predict/multiclass")
async def predict_multiclass(file: UploadFile = File(...)):
    if modelo_mc is None:
        return JSONResponse(status_code=400, content={"error": "Não foi possível encontrar o modelo multiclasse"})

    df = pd.read_csv(file.file, sep=None, engine="python")
    if "id" not in df.columns:
        return JSONResponse(status_code=400, content={"error": "O dataset precisa ter a coluna 'id' "})

    ids = df["id"].astype(str)
    X = df.drop(columns=["id"], errors="ignore")
    y_pred = modelo_mc.predict(X)

    out = pd.DataFrame({"id": ids, "classe_falha_pred": y_pred})
    out_csv = os.path.join(BASE_DIR, "predicoes_multiclasse.csv")
    out.to_csv(out_csv, index=False)
    return FileResponse(out_csv, filename="predicoes_multiclasse.csv")
