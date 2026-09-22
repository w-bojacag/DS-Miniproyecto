import streamlit as st
import pandas as pd
import requests
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuración de la página
st.set_page_config(page_title="Scoring de Crédito", page_icon="🏦", layout="centered")

st.title("🏦 Sistema de Predicción de Riesgo Crediticio")
st.write("""
Esta herramienta utiliza el modelo de **Regresión Logística (Balanceada)** para predecir
la probabilidad de que un cliente presente mora grave en los próximos 2 años.
""")

# Crear columnas para organizar el formulario
col1, col2 = st.columns(2)

with col1:
    st.subheader("Datos Demográficos y Financieros")
    edad = st.number_input("Edad del solicitante", min_value=18, max_value=100, value=35)
    ingreso_mensual = st.number_input("Ingreso Mensual ($)", min_value=0, max_value=500000, value=5000)
    dependientes = st.number_input("Número de dependientes", min_value=0, max_value=20, value=0)
    razon_deuda = st.number_input("Razón de Deuda (Obligaciones / Ingreso)", min_value=0.0, max_value=100.0, value=0.3)

with col2:
    st.subheader("Comportamiento e Historial")
    utilizacion = st.number_input("Utilización del límite de crédito (0 a 1)", min_value=0.0, max_value=10.0, value=0.5)
    lineas_abiertas = st.number_input("Líneas de crédito abiertas", min_value=0, max_value=50, value=5)
    creditos_inmob = st.number_input("Créditos hipotecarios", min_value=0, max_value=15, value=1)

    st.markdown("**Historial de Moras Previas (veces)**")
    moras_30_59 = st.number_input("Moras de 30-59 días", min_value=0, max_value=20, value=0)
    moras_60_89 = st.number_input("Moras de 60-89 días", min_value=0, max_value=20, value=0)
    moras_90 = st.number_input("Moras de 90+ días", min_value=0, max_value=20, value=0)

# Botón de predicción
if st.button("Evaluar Solicitante", type="primary"):
    input_data = {
        "edad": edad,
        "ingreso_mensual": ingreso_mensual,
        "dependientes": dependientes,
        "razon_deuda": razon_deuda,
        "utilizacion": utilizacion,
        "lineas_abiertas": lineas_abiertas,
        "creditos_inmob": creditos_inmob,
        "moras_30_59": moras_30_59,
        "moras_60_89": moras_60_89,
        "moras_90": moras_90
    }

    try:
        api_url = os.getenv("API_URL", "http://localhost")
        api_port = os.getenv("API_PORT", "8000")
        api_endpoint = os.getenv("API_ENDPOINT", "/api/v1/predict")

        full_url = f"{api_url}:{api_port}{api_endpoint}"

        response = requests.post(full_url, json=input_data, timeout=10)
        response.raise_for_status()
        resultado = response.json()

        probabilidad = resultado.get("probabilidad_incumplimiento", 0)
        clasificacion = resultado.get("clasificacion", "")
        explicacion = resultado.get("explicacion", {})
        variables = explicacion.get("variables", [])

        st.divider()
        st.subheader("Resultado de la Evaluación")
        st.progress(probabilidad)

        if clasificacion.upper() == "RIESGO ALTO":
            st.error(f"⚠️ {clasificacion.upper()}: Probabilidad de incumplimiento del {probabilidad*100:.1f}%")
            st.write("Recomendación: **Revisión Manual / Rechazar**")
        else:
            st.success(f"✅ {clasificacion.upper()}: Probabilidad de incumplimiento del {probabilidad*100:.1f}%")
            st.write("Recomendación: **Aprobar**")

        if variables:
            st.subheader("📊 Explicación del Modelo")
            st.write(f"*Unidad de aporte: {explicacion.get('unidad_aporte', 'N/A')}*")

            col_var, col_valor, col_aporte, col_dir = st.columns(4)
            with col_var:
                st.write("**Variable**")
            with col_valor:
                st.write("**Valor**")
            with col_aporte:
                st.write("**Aporte**")
            with col_dir:
                st.write("**Dirección**")

            for var in variables:
                col_var, col_valor, col_aporte, col_dir = st.columns(4)
                with col_var:
                    st.write(f"{var.get('nombre', var.get('variable'))}")
                with col_valor:
                    st.write(f"{var.get('valor_original', 'N/A')}")
                with col_aporte:
                    aporte_val = var.get('aporte', 0)
                    color = "🟢" if aporte_val < 0 else "🔴"
                    st.write(f"{color} {aporte_val:+.2f}")
                with col_dir:
                    st.write(var.get('direccion', 'N/A'))

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error al conectar con la API: {str(e)}")
    except Exception as e:
        st.error(f"❌ Error procesando la respuesta: {str(e)}")
