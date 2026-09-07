import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    input_data = pd.DataFrame([{
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
    }])
    
    try:
        # Intenta cargar el modelo real
        modelo = joblib.load("src/modelo_logreg.pkl")
    except FileNotFoundError:
        # Si no existe, crea un modelo simulado temporal para pruebas de interfaz
        modelo = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression())
        ])
        # Creamos dos filas dummy con ambas clases (0 y 1) para evitar el error del solver
        X_dummy = pd.DataFrame([
            [35, 5000, 0, 0.3, 0.5, 5, 1, 0, 0, 0],
            [50, 2000, 1, 0.8, 0.9, 2, 0, 1, 0, 0]
        ], columns=input_data.columns)
        y_dummy = [0, 1]
        modelo.fit(X_dummy, y_dummy)
    
    probabilidad = float(modelo.predict_proba(input_data)[0][1])
    
    st.divider()
    st.subheader("Resultado de la Evaluación")
    st.progress(probabilidad)
    
    if probabilidad >= 0.5:
        st.error(f"⚠️ ALTO RIESGO: Probabilidad de incumplimiento del {probabilidad*100:.1f}%")
        st.write("Recomendación: **Revisión Manual / Rechazar**")
    else:
        st.success(f"✅ RIESGO CONTROLADO: Probabilidad de incumplimiento del {probabilidad*100:.1f}%")
        st.write("Recomendación: **Aprobar**")