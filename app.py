import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Configuración de la página web
st.set_page_config(page_title="Predicción de Accidentes Viales", page_icon="🚗", layout="centered")

st.title("🚗 Simulador de Siniestralidad Vial (Machine Learning)")
st.write("Modifica los factores de conducción para predecir la tasa de siniestros mortales por billón de millas.")

# 1. Cargar el escalador y el modelo generados en tu código
@st.cache_resource # Evita recargar los archivos en cada clic
def cargar_artefactos():
    with open('scaler_car_crashes.pkl', 'rb') as f_scaler:
        scaler = pickle.load(f_scaler)
    with open('model_car_crashes.pkl', 'rb') as f_model:
        model = pickle.load(f_model)
    return scaler, model

try:
    scaler, model = cargar_artefactos()
except FileNotFoundError:
    st.error("Por favor, asegúrate de haber ejecutado tu script para generar 'scaler_car_crashes.pkl' y 'model_car_crashes.pkl'")

# 2. Crear la interfaz gráfica en la barra lateral o panel principal
st.header("📊 Parámetros del Conductor")

col1, col2 = st.columns(2)

with col1:
    speeding = st.slider("Porcentaje por Exceso de Velocidad", 0.0, 100.0, 30.0)
    alcohol = st.slider("Porcentaje por Consumo de Alcohol", 0.0, 100.0, 25.0)
    not_distracted = st.slider("Porcentaje de NO Distraídos", 0.0, 100.0, 80.0)

with col2:
    no_previous = st.slider("Porcentaje Sin Accidentes Previos", 0.0, 100.0, 90.0)
    ins_premium = st.number_input("Costo de Prima de Seguro ($)", min_value=500.0, max_value=2000.0, value=800.0)
    ins_losses = st.number_input("Pérdidas de la Aseguradora ($)", min_value=50.0, max_value=300.0, value=130.0)

# 3. Procesar los datos de entrada cuando el usuario presione el botón
if st.button("🔮 Calcular Predicción"):
    
    # Crear el diccionario con el orden EXACTO de las columnas de entrenamiento de tu X
    datos_usuario = {
        'speeding': speeding,
        'alcohol': alcohol,
        'not_distracted': not_distracted,
        'no_previous': no_previous,
        'ins_premium': ins_premium,
        'ins_losses': ins_losses
    }
    
    df_entrada = pd.DataFrame([datos_usuario])
    
    # Aplicar la estandarización tal cual lo hiciste en tu Fase 3 (¡importante usar transform!)
    datos_escalados = scaler.transform(df_entrada)
    
    # Realizar la predicción con el modelo ganador (Random Forest)
    prediccion = model.predict(datos_escalados)
    
    # Mostrar el resultado de manera llamativa
    st.success(f"### 📈 Resultado de la Predicción:")
    st.metric(label="Siniestros mortales estimados (por billón de millas)", value=f"{prediccion[0]:.2f}")