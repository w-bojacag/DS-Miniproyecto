import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_URL = os.getenv("API_URL", "http://localhost")
API_PORT = os.getenv("API_PORT", "8000")
API_ENDPOINT = os.getenv("API_ENDPOINT", "/api/v1/predict")
CARTERA_PATH = Path(__file__).resolve().parent / "data" / "cartera_agregada.json"

ORDEN = {
    "grupo_edad": ["18-30", "31-40", "41-50", "51-60", "61-70", "70+"],
    "tramo_utilizacion": ["0-10%", "10-25%", "25-50%", "50-75%", "75-100%", ">100%"],
    "quintil_ingreso": ["Q1 (más bajo)", "Q2", "Q3", "Q4", "Q5 (más alto)", "No reportado"],
    "moras_90_previas": ["0", "1", "2", "3 o más"],
}

st.set_page_config(page_title="Scoring de Crédito", page_icon="🏦", layout="wide")

st.title("🏦 Sistema de Predicción de Riesgo Crediticio")
st.write(
    "Estima la probabilidad de que un solicitante presente mora grave (90 días o más) "
    "en los próximos 2 años, con un modelo **XGBoost** servido por la API, y muestra "
    "los patrones de incumplimiento de la cartera histórica."
)

tab_eval, tab_cartera = st.tabs(["🧾 Evaluar solicitante", "📊 Cartera histórica"])


# ---------------------------------------------------------------- Evaluación
def grafica_explicacion(variables: list) -> alt.Chart:
    df = pd.DataFrame(variables)
    df["etiqueta"] = df.apply(
        lambda r: f"{r['nombre']} = {r['valor_original']}"
        if r.get("valor_original") is not None else f"{r['nombre']} (sin dato)", axis=1)
    df["efecto"] = df["aporte"].apply(lambda a: "Aumenta el riesgo" if a > 0 else "Reduce el riesgo")
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("aporte:Q", title="Aporte a la predicción (log-odds)"),
            y=alt.Y("etiqueta:N", sort=alt.EncodingSortField(field="aporte", op="sum",
                                                            order="descending"), title=None),
            color=alt.Color("efecto:N", title=None,
                            scale=alt.Scale(domain=["Aumenta el riesgo", "Reduce el riesgo"],
                                            range=["#c0392b", "#27ae60"])),
            tooltip=[alt.Tooltip("nombre:N", title="Variable"),
                     alt.Tooltip("valor_original:Q", title="Valor"),
                     alt.Tooltip("aporte:Q", title="Aporte", format="+.3f")],
        )
        .properties(height=60 * len(df))
    )


with tab_eval:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Datos Demográficos y Financieros")
        edad = st.number_input("Edad del solicitante", min_value=18, max_value=100, value=35)
        reporta_ingreso = st.checkbox("El solicitante reporta ingreso", value=True)
        ingreso_mensual = st.number_input("Ingreso Mensual ($)", min_value=0, max_value=500000,
                                          value=5000, disabled=not reporta_ingreso)
        dependientes = st.number_input("Número de dependientes", min_value=0, max_value=20, value=0)
        razon_deuda = st.number_input("Razón de Deuda (Obligaciones / Ingreso)", min_value=0.0,
                                      max_value=100.0, value=0.3)

    with col2:
        st.subheader("Comportamiento e Historial")
        utilizacion = st.number_input("Utilización del límite de crédito (1 = 100 %)",
                                      min_value=0.0, max_value=10.0, value=0.5)
        lineas_abiertas = st.number_input("Líneas de crédito abiertas", min_value=0, max_value=50,
                                          value=5)
        creditos_inmob = st.number_input("Créditos hipotecarios", min_value=0, max_value=15, value=1)

        st.markdown("**Historial de Moras Previas (veces)**")
        moras_30_59 = st.number_input("Moras de 30-59 días", min_value=0, max_value=20, value=0)
        moras_60_89 = st.number_input("Moras de 60-89 días", min_value=0, max_value=20, value=0)
        moras_90 = st.number_input("Moras de 90+ días", min_value=0, max_value=20, value=0)

    if st.button("Evaluar Solicitante", type="primary"):
        input_data = {
            "edad": edad,
            "ingreso_mensual": ingreso_mensual if reporta_ingreso else None,
            "dependientes": dependientes,
            "razon_deuda": razon_deuda,
            "utilizacion": utilizacion,
            "lineas_abiertas": lineas_abiertas,
            "creditos_inmob": creditos_inmob,
            "moras_30_59": moras_30_59,
            "moras_60_89": moras_60_89,
            "moras_90": moras_90,
        }

        try:
            full_url = f"{API_URL}:{API_PORT}{API_ENDPOINT}"
            response = requests.post(full_url, json=input_data, timeout=10)
            response.raise_for_status()
            resultado = response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error al conectar con la API: {str(e)}")
        else:
            probabilidad = resultado.get("probabilidad_incumplimiento", 0)
            umbral = resultado.get("umbral_aplicado")
            clasificacion = resultado.get("clasificacion", "")
            variables = resultado.get("explicacion", {}).get("variables", [])

            st.divider()
            st.subheader("Resultado de la Evaluación")
            m1, m2 = st.columns(2)
            m1.metric("Probabilidad de incumplimiento", f"{probabilidad * 100:.1f} %")
            if umbral is not None:
                m2.metric("Umbral de decisión", f"{umbral * 100:.0f} %")
            st.progress(min(float(probabilidad), 1.0))

            if clasificacion.upper() == "RIESGO ALTO":
                st.error(f"⚠️ {clasificacion.upper()}: la probabilidad supera el umbral de decisión.")
                st.write("Recomendación: **Revisión Manual / Rechazar**")
            else:
                st.success(f"✅ {clasificacion.upper()}: la probabilidad está por debajo del umbral.")
                st.write("Recomendación: **Aprobar**")

            if variables:
                st.subheader("📊 ¿Qué explica esta predicción?")
                st.altair_chart(grafica_explicacion(variables), width="stretch")
                st.caption(
                    "Principales factores según SHAP. Las barras rojas empujan la predicción "
                    "hacia el incumplimiento; las verdes la alejan. La longitud indica cuánto "
                    "pesa cada factor para este solicitante en particular."
                )


# ---------------------------------------------------------------- Cartera
@st.cache_data
def cargar_cartera(path: Path):
    if not path.exists():
        return None
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))


def tasa_por(df: pd.DataFrame, dim: str) -> alt.Chart:
    g = df.groupby(dim)[["clientes", "incumplidos"]].sum().reset_index()
    g["tasa"] = 100 * g["incumplidos"] / g["clientes"]
    return (
        alt.Chart(g)
        .mark_bar(color="#2c5d8f")
        .encode(
            x=alt.X(f"{dim}:N", sort=ORDEN[dim], title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("tasa:Q", title="% incumplimiento"),
            tooltip=[alt.Tooltip(f"{dim}:N", title="Segmento"),
                     alt.Tooltip("tasa:Q", title="% incumplimiento", format=".2f"),
                     alt.Tooltip("clientes:Q", title="Clientes", format=",")],
        )
        .properties(height=260)
    )


with tab_cartera:
    cartera = cargar_cartera(CARTERA_PATH)
    if cartera is None:
        st.warning("No se encontró la cartera agregada. Genérela con `python src/build_cartera.py`.")
    else:
        st.write("Patrones de incumplimiento en la cartera histórica. Use los filtros para "
                 "analizar un segmento específico.")
        f1, f2, f3, f4 = st.columns(4)
        filtros = {
            "grupo_edad": f1.multiselect("Grupo de edad", ORDEN["grupo_edad"]),
            "quintil_ingreso": f2.multiselect("Ingreso", ORDEN["quintil_ingreso"]),
            "tramo_utilizacion": f3.multiselect("Utilización del cupo", ORDEN["tramo_utilizacion"]),
            "moras_90_previas": f4.multiselect("Moras previas 90+ días", ORDEN["moras_90_previas"]),
        }
        seg = cartera
        for dim, valores in filtros.items():
            if valores:
                seg = seg[seg[dim].isin(valores)]

        total_global = cartera["clientes"].sum()
        tasa_global = 100 * cartera["incumplidos"].sum() / total_global
        n_seg = int(seg["clientes"].sum())

        k1, k2, k3 = st.columns(3)
        k1.metric("Clientes en el segmento", f"{n_seg:,}".replace(",", "."))
        if n_seg:
            tasa_seg = 100 * seg["incumplidos"].sum() / n_seg
            k2.metric("Tasa de incumplimiento", f"{tasa_seg:.2f} %",
                      delta=f"{tasa_seg - tasa_global:+.2f} pp vs. cartera total",
                      delta_color="inverse")
        k3.metric("Tasa de la cartera total", f"{tasa_global:.2f} %")

        if n_seg == 0:
            st.info("No hay clientes con esa combinación de filtros.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Por grupo de edad**")
                st.altair_chart(tasa_por(seg, "grupo_edad"), width="stretch")
                st.markdown("**Por moras previas de 90+ días**")
                st.altair_chart(tasa_por(seg, "moras_90_previas"), width="stretch")
            with c2:
                st.markdown("**Por utilización del cupo**")
                st.altair_chart(tasa_por(seg, "tramo_utilizacion"), width="stretch")
                st.markdown("**Por nivel de ingreso**")
                st.altair_chart(tasa_por(seg, "quintil_ingreso"), width="stretch")
            st.caption("Incumplimiento: mora de 90 días o más en los 2 años siguientes. "
                       "Segmentos con pocos clientes pueden mostrar tasas inestables.")
