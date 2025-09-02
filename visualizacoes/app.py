import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Dashboard - Manutenção Preditiva", page_icon="🛠️", layout="wide")
st.title("🔧 Sistema de Manutenção Preditiva")
st.markdown("Compare modelos, visualize métricas e gere o CSV para a API do Bootcamp CDIA.")

BASE_PATH = os.path.dirname(__file__)

def load_metrics(path: str):
    """Carrega o JSON e retorna (results_dict, best_model_name) de forma robusta,
    aceitando tanto o formato novo (com 'all_results') quanto o antigo."""
    if not os.path.exists(path):
        return {}, None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "all_results" in data:
        return data.get("all_results", {}), data.get("best_model")

    if isinstance(data, dict):
        return data, None

    return {}, None


def safe_number(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def barplot_from_mapping(mapping: dict, ylabel: str, title: str):
    if not mapping:
        st.info("Não há valores para exibir.")
        return
    fig, ax = plt.subplots()
    ax.bar(mapping.keys(), mapping.values())
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticklabels(mapping.keys(), rotation=15, ha="right")
    st.pyplot(fig)


aba_binaria, aba_multirrotulo, aba_multiclasse = st.tabs(["🟩 Binária (FTE)", "🟨 Multirrótulo", "🟥 Multiclasse"])

with aba_binaria:
    st.header("📊 Modelagem Binária (FTE)")
    metricas_path = os.path.join(BASE_PATH, "metricas_binario.json")
    results, best_model = load_metrics(metricas_path)

    if results:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔎 AUC-ROC")
            if results:
                for m, d in results.items():
                    if not isinstance(d, dict):
                        st.metric(m, "—")
                        continue
                    auc = safe_number(d.get("roc_auc"))
                    st.metric(m, f"{auc:.4f}" if auc is not None else "—")
            else:
                st.info("Sem valores de ROC AUC.")

        with col2:
            st.subheader("📋 F1-Score (classe 1)")
            if results:
                for m, d in results.items():
                    if not isinstance(d, dict):
                        st.metric(m, "—")
                        continue
                    rep = d.get("classification_report", {})
                    cls1 = rep.get("1", {}) if isinstance(rep, dict) else {}
                    f1 = safe_number(cls1.get("f1-score"))
                    st.metric(m, f"{f1:.4f}" if f1 is not None else "—")
            else:
                st.info("Sem F1 (classe 1) disponível.")

        st.markdown("### 📈 Comparação gráfica (F1 / Recall / Precision)")
        metrica = st.selectbox("Métrica:", ["f1-score", "recall", "precision"], key="bin")
        vals = {}
        for m, d in results.items():
            if not isinstance(d, dict):
                continue
            rep = d.get("classification_report", {})
            cls1 = rep.get("1", {}) if isinstance(rep, dict) else {}
            num = safe_number(cls1.get(metrica))
            if num is not None:
                vals[m] = num
        barplot_from_mapping(vals, ylabel=metrica.title(), title=f"{metrica.title()} (classe 1 - FTE)")

        if best_model:
            st.caption(f"🏆 Melhor modelo (treino): **{best_model}**")
    else:
        st.info("Execute o script de **treinamento binário** para gerar `visualizacoes/metricas_binario.json`.")

with aba_multirrotulo:
    st.header("📊 Modelagem Multirrótulo (todas as falhas)")
    path = os.path.join(BASE_PATH, "metricas_multirrotulo.json")
    results, best_model = load_metrics(path)

    if results:
        df_table = {}
        for m, d in results.items():
            if not isinstance(d, dict):
                continue
            df_table[m] = {
                "F1 Micro": round(safe_number(d.get("f1_micro"), 0.0), 4),
                "F1 Macro": round(safe_number(d.get("f1_macro"), 0.0), 4),
                "Precision": round(safe_number(d.get("precision_micro"), 0.0), 4),
                "Recall": round(safe_number(d.get("recall_micro"), 0.0), 4),
                "Hamming Loss": round(safe_number(d.get("hamming_loss"), 0.0), 4),
                "Subset Accuracy": round(safe_number(d.get("subset_accuracy"), 0.0), 4),
            }

        if df_table:
            st.dataframe(pd.DataFrame(df_table).T)
            st.markdown("### 📈 Gráfico comparativo")
            metrica = st.selectbox("Métrica:", ["F1 Micro", "F1 Macro", "Precision", "Recall", "Subset Accuracy"], key="multi")
            vals = {m: df_table[m][metrica] for m in df_table}
            barplot_from_mapping(vals, ylabel=metrica, title=f"Comparação de {metrica}")
        else:
            st.info("Não há métricas válidas para exibir.")

        if best_model:
            st.caption(f"🏆 Melhor modelo (treino): **{best_model}**")
    else:
        st.info("Execute o script de **treinamento multirrótulo** para gerar `visualizacoes/metricas_multirrotulo.json`.")

    st.markdown("---")
    st.markdown("## 🆕 Gerar predições para **novos dados** da máquina")
    st.markdown("Envie um CSV com o mesmo esquema do `bootcamp_test.csv` (inclui `id`, `id_produto`, `tipo`, variáveis de processo etc.).")

    novo_csv = st.file_uploader("📤 Envie novos dados (CSV)", type=["csv"], key="uploader_multirrotulo_novos")
    if novo_csv is not None:
        import joblib
        import numpy as np
        import requests

        modelo_path = os.path.join(BASE_PATH, "..", "api", "model_multirrotulo.pkl")
        if not os.path.exists(modelo_path):
            st.error("Modelo multirrótulo não encontrado. Execute o treinamento multirrótulo para salvar o modelo.")
        else:
            try:
                modelo = joblib.load(modelo_path)
                dados = pd.read_csv(novo_csv)
            except Exception as e:
                st.error(f"Erro ao carregar modelo ou CSV: {e}")
            else:
                if "id" not in dados.columns:
                    st.error("O CSV precisa conter a coluna 'id'.")
                else:
                    ids = dados["id"]
                    X_novos = dados.drop(columns=["id"])

                    try:
                        y_pred = modelo.predict(X_novos)
                        cols = ["FDF", "FDC", "FP", "FTE", "FA"][:np.array(y_pred).shape[1]]
                        df_out = pd.DataFrame(y_pred, columns=cols)
                        df_out.insert(0, "id", ids)
                        st.success("✅ Arquivo de predições gerado!")
                        st.download_button(
                            "📥 Baixar predicoes_para_api.csv",
                            df_out.to_csv(index=False).encode("utf-8"),
                            file_name="predicoes_para_api.csv",
                            mime="text/csv",
                        )

                        st.markdown("### 🛰️ Enviar direto para a API oficial")
                        with st.form("form_envio_api"):
                            token = st.text_input("X-API-Key (token)", type="password")
                            threshold = st.number_input("threshold (0–1)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
                            enviar = st.form_submit_button("🚀 Enviar para API /evaluate/multilabel_metrics")

                        if enviar:
                            files = {"file": ("predicoes_para_api.csv", df_out.to_csv(index=False), "text/csv")}
                            headers = {"X-API-Key": token} if token else {}
                            params = {"threshold": threshold}
                            try:
                                resp = requests.post(
                                    "http://34.193.187.218:5000/evaluate/multilabel_metrics",
                                    headers=headers, files=files, params=params, timeout=60
                                )
                                if resp.status_code == 200:
                                    st.success("✅ Avaliação realizada!")
                                    st.json(resp.json())
                                else:
                                    st.error(f"Erro {resp.status_code}: {resp.text[:300]}")
                            except Exception as e:
                                st.error(f"Erro na requisição: {e}")
                    except Exception as e:
                        st.error(f"Erro ao prever: {e}")

with aba_multiclasse:
    st.header("📊 Modelagem Multiclasse (falha dominante)")
    path = os.path.join(BASE_PATH, "metricas_multiclasse.json")
    results, best_model = load_metrics(path)

    if results:
        df_table = {}
        for m, d in results.items():
            if not isinstance(d, dict):
                continue
            df_table[m] = {
                "Accuracy": round(safe_number(d.get("accuracy"), 0.0), 4),
                "Balanced Accuracy": round(safe_number(d.get("balanced_accuracy"), 0.0), 4),
                "F1 Macro": round(safe_number(d.get("f1_macro"), 0.0), 4),
            }

        if df_table:
            st.dataframe(pd.DataFrame(df_table).T)
            st.markdown("### 📈 Gráfico comparativo")
            metrica = st.selectbox("Métrica:", ["Accuracy", "Balanced Accuracy", "F1 Macro"], key="multi_cls")
            vals = {m: df_table[m][metrica] for m in df_table}
            barplot_from_mapping(vals, ylabel=metrica, title=f"Comparação de {metrica}")
        else:
            st.info("Não há métricas válidas para exibir.")

        if best_model:
            st.caption(f"🏆 Melhor modelo (treino): **{best_model}**")
    else:
        st.info("Execute o script de **treinamento multiclasse** para gerar `visualizacoes/metricas_multiclasse.json`.")

st.markdown("---")
st.markdown("Desenvolvido para o Bootcamp CDIA (SENAI-SC) – Manutenção Preditiva.")
