import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Predicción de Siniestralidad Vial",
    page_icon="🚗",
    layout="wide"
)

# ── Estilos ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #e94560;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        color: #e94560;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0b0;
        margin-top: 0.3rem;
    }
    .info-box {
        background: #1a1a2e;
        border-left: 4px solid #0f3460;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #a0a0b0;
    }
</style>
""", unsafe_allow_html=True)

# ── Pipeline de entrenamiento (se ejecuta una sola vez por sesión) ───────────
@st.cache_resource
def entrenar_pipeline():
    """
    Carga el dataset, entrena los 4 modelos y devuelve el mejor junto
    con el scaler, las métricas y los valores reales min/max por columna.
    """
    df = sns.load_dataset('car_crashes').set_index('abbrev')

    X = df.drop(columns=['total'])
    y = df['total']

    # Guardar rangos REALES del dataset para usarlos en los sliders
    stats = X.describe().loc[['min', 'max', 'mean', 'std']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    modelos = {
        "Regresión Lineal":   LinearRegression(),
        "Árbol de Decisión":  DecisionTreeRegressor(random_state=42, max_depth=4),
        "Random Forest":      RandomForestRegressor(random_state=42, n_estimators=200,
                                                     max_depth=5, min_samples_leaf=2),
        "Gradient Boosting":  GradientBoostingRegressor(random_state=42, n_estimators=200,
                                                         learning_rate=0.05, max_depth=3,
                                                         subsample=0.8),
    }

    resultados = []
    entrenados = {}
    for nombre, m in modelos.items():
        m.fit(X_train_sc, y_train)
        entrenados[nombre] = m
        y_pred = m.predict(X_test_sc)
        y_pred_tr = m.predict(X_train_sc)
        resultados.append({
            "Modelo":      nombre,
            "MAE":         round(mean_absolute_error(y_test, y_pred), 4),
            "RMSE":        round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "R² Test":     round(r2_score(y_test, y_pred), 4),
            "R² Train":    round(r2_score(y_train, y_pred_tr), 4),
        })

    df_res = pd.DataFrame(resultados).sort_values("RMSE").reset_index(drop=True)

    # El mejor modelo según RMSE en test
    mejor_nombre = df_res.iloc[0]["Modelo"]
    mejor_modelo = entrenados[mejor_nombre]

    return scaler, entrenados, mejor_nombre, mejor_modelo, df_res, stats, X.columns.tolist(), df

# ── Carga ────────────────────────────────────────────────────────────────────
with st.spinner("Entrenando modelos sobre el dataset real..."):
    scaler, entrenados, mejor_nombre, modelo_activo, df_res, stats, feature_cols, df_full = entrenar_pipeline()

# ── Título ───────────────────────────────────────────────────────────────────
st.title("🚗 Simulador de Siniestralidad Vial · Machine Learning")
st.caption("Predicción de la tasa de siniestros mortales por billón de millas recorridas — dataset *car_crashes* (EE.UU.)")

st.divider()

# ── Layout principal ─────────────────────────────────────────────────────────
col_controles, col_resultado = st.columns([3, 2], gap="large")

with col_controles:
    st.subheader("⚙️ Parámetros de entrada")

    # Selector de modelo
    modelo_seleccionado = st.selectbox(
        "Algoritmo de predicción",
        options=list(entrenados.keys()),
        index=list(entrenados.keys()).index(mejor_nombre),
        help=f"El modelo con menor RMSE en test es **{mejor_nombre}**"
    )
    modelo_activo = entrenados[modelo_seleccionado]

    st.markdown("---")

    # ── Sliders con rangos REALES del dataset ────────────────────────────────
    def slider_real(label, col, help_txt=""):
        vmin  = float(stats.loc['min', col])
        vmax  = float(stats.loc['max', col])
        vmean = float(stats.loc['mean', col])
        return st.slider(
            label,
            min_value=round(vmin, 2),
            max_value=round(vmax, 2),
            value=round(vmean, 2),
            step=round((vmax - vmin) / 100, 3),
            help=help_txt
        )

    c1, c2 = st.columns(2)
    with c1:
        speeding       = slider_real("🚀 % por Exceso de Velocidad", "speeding",
                                     "Porcentaje de conductores involucrados en siniestros fatales que excedían la velocidad permitida.")
        alcohol        = slider_real("🍺 % por Consumo de Alcohol",  "alcohol",
                                     "Porcentaje de conductores involucrados que tenían alcohol en sangre sobre el límite legal.")
        not_distracted = slider_real("📵 % NO Distraídos",           "not_distracted",
                                     "Porcentaje de conductores que NO estaban distraídos al momento del accidente.")
    with c2:
        no_previous    = slider_real("✅ % Sin Accidentes Previos",  "no_previous",
                                     "Porcentaje de conductores sin historial previo de accidentes.")
        ins_premium    = st.number_input(
            "💳 Prima de Seguro ($/año)",
            min_value=float(stats.loc['min', 'ins_premium']),
            max_value=float(stats.loc['max', 'ins_premium']),
            value=float(stats.loc['mean', 'ins_premium']),
            step=10.0,
            help="Costo promedio anual de la póliza de seguro de auto en ese estado."
        )
        ins_losses     = st.number_input(
            "📉 Pérdidas de Aseguradora ($/año)",
            min_value=float(stats.loc['min', 'ins_losses']),
            max_value=float(stats.loc['max', 'ins_losses']),
            value=float(stats.loc['mean', 'ins_losses']),
            step=5.0,
            help="Pérdidas promedio que las aseguradoras pagan por asegurado en ese estado."
        )

    predecir = st.button("🔮 Calcular Predicción", type="primary", use_container_width=True)

# ── Panel de resultado ───────────────────────────────────────────────────────
with col_resultado:
    st.subheader("📈 Resultado")

    if predecir:
        entrada = pd.DataFrame([{
            'speeding':       speeding,
            'alcohol':        alcohol,
            'not_distracted': not_distracted,
            'no_previous':    no_previous,
            'ins_premium':    ins_premium,
            'ins_losses':     ins_losses,
        }])[feature_cols]  # Garantiza el orden correcto de columnas

        entrada_sc  = scaler.transform(entrada)
        prediccion  = modelo_activo.predict(entrada_sc)[0]

        # Color semáforo según severidad
        p25 = df_full['total'].quantile(0.25)
        p75 = df_full['total'].quantile(0.75)
        if prediccion <= p25:
            color, nivel = "#27ae60", "🟢 Bajo"
        elif prediccion <= p75:
            color, nivel = "#f39c12", "🟡 Moderado"
        else:
            color, nivel = "#e74c3c", "🔴 Alto"

        st.markdown(f"""
        <div class="metric-box">
            <div style="color:{color}; font-size:3.5rem; font-weight:900; line-height:1">
                {prediccion:.2f}
            </div>
            <div class="metric-label">siniestros mortales por billón de millas</div>
            <div style="margin-top:0.8rem; font-size:1rem; font-weight:600; color:{color}">
                Nivel de riesgo: {nivel}
            </div>
            <div class="metric-label" style="margin-top:0.4rem">
                Modelo: {modelo_seleccionado}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Contexto: distribución del dataset
        st.caption("**Distribución histórica del dataset (referencia)**")
        q_data = {
            "Percentil": ["P25 (bajo)", "P50 (mediana)", "P75 (alto)"],
            "Total siniestros": [
                round(df_full['total'].quantile(0.25), 2),
                round(df_full['total'].quantile(0.50), 2),
                round(df_full['total'].quantile(0.75), 2),
            ]
        }
        st.dataframe(pd.DataFrame(q_data), hide_index=True, use_container_width=True)

    else:
        st.info("Ajusta los parámetros y presiona **Calcular Predicción** para ver el resultado.")

        # Mostrar importancia de features del mejor modelo si tiene el atributo
        if hasattr(modelo_activo, 'feature_importances_'):
            st.markdown("---")
            st.caption(f"**Importancia de variables — {modelo_seleccionado}**")
            imp = pd.DataFrame({
                "Variable":    feature_cols,
                "Importancia": modelo_activo.feature_importances_
            }).sort_values("Importancia", ascending=False)
            st.bar_chart(imp.set_index("Variable"))

# ── Tabla comparativa de modelos ─────────────────────────────────────────────
st.divider()
with st.expander("📊 Comparativa de modelos (métricas en conjunto de prueba)", expanded=False):
    st.dataframe(
        df_res.style.highlight_min(subset=["MAE", "RMSE"], color="#1a4a1a")
                    .highlight_max(subset=["R² Test"], color="#1a4a1a"),
        use_container_width=True,
        hide_index=True
    )
    st.caption("✅ Verde = mejor valor en esa métrica. El modelo con menor RMSE se selecciona como predeterminado.")