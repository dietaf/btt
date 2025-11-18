
import streamlit as st
import sqlite3
import time
import os
from datetime import datetime
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# ===================== CONFIGURACIÓN DE BASE DE DATOS =====================
DB_NAME = "trading_logs.db"

def init_database():
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            mensaje TEXT
        )
        """)
        conn.commit()
        conn.close()
    except sqlite3.DatabaseError:
        if os.path.exists(DB_NAME):
            os.remove(DB_NAME)
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            mensaje TEXT
        )
        """)
        conn.commit()
        conn.close()

init_database()

def log_event(mensaje):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO logs (timestamp, mensaje) VALUES (?, ?)", (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), mensaje))
    conn.commit()
    conn.close()

def get_logs(limit=100):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT timestamp, mensaje FROM logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

def clear_logs():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM logs")
    conn.commit()
    conn.close()

# ===================== AUTENTICACIÓN =====================
st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")
st.title("🔐 Acceso al Bot de Trading")
password = st.text_input("Ingrese la contraseña:", type="password")
if password != "admin123":
    st.warning("Por favor ingrese la contraseña correcta para acceder al dashboard.")
    st.stop()

# ===================== INTERFAZ PRINCIPAL =====================
st.title("📈 Dashboard Profesional - Bot de Trading con ML y PnL Real")

# Panel de configuración
st.sidebar.header("Configuración")
strategy = st.sidebar.selectbox("Estrategia", ["TrendShift", "MeanReversion", "Breakout"])
pair = st.sidebar.selectbox("Par", ["EURUSD=X", "GBPUSD=X", "BTC-USD"])
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m"])

st.sidebar.subheader("Parámetros Iniciales")
volumen = st.sidebar.number_input("Volumen ($)", min_value=1000, max_value=100000, value=10000)
risk = st.sidebar.slider("Riesgo Mínimo (%)", 0, 100, 5)
confianza_minima = st.sidebar.slider("Confianza Mínima (%)", 0, 100, 10)
probabilidad_minima = st.sidebar.slider("Probabilidad ML mínima", 0.0, 1.0, 0.6, 0.01)

col1, col2 = st.sidebar.columns(2)
iniciar = col1.button("INICIAR")
detener = col2.button("DETENER")

# ===================== ENTRENAR MODELO ML =====================
st.subheader("Entrenando modelo ML con optimización...")
progress_bar = st.progress(0)
status_text = st.empty()

hist_data = yf.download(pair, period="30d", interval="5m")
if not hist_data.empty:
    hist_data['Return'] = hist_data['Close'].pct_change()
    hist_data['Signal'] = np.where(hist_data['Return'] > 0, 1, 0)
    hist_data.dropna(inplace=True)
    X = hist_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = hist_data['Signal']

    param_grid = {
        'n_estimators': [50],
        'max_depth': [5, None],
        'min_samples_split': [2]
    }

    status_text.text("Ejecutando GridSearchCV...")
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=2, scoring='accuracy')
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    model = grid_search.best_estimator_

    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.01)

    status_text.text(f"Entrenamiento completado ✅ | Mejores parámetros: {best_params}")
else:
    st.error("No se pudieron obtener datos históricos para entrenar el modelo.")

# ===================== DASHBOARD EN TIEMPO REAL =====================
placeholder_chart = st.empty()
placeholder_metrics = st.empty()

precios = []
signales = []
confianzas = []
trades_ejecutados = 0
ganancia_total = 0.0
pnl_por_trade = []
posicion_abierta = False
precio_entrada = None

if iniciar:
    st.success("Bot iniciado")
    log_event(f"Bot iniciado con estrategia {strategy} en {pair}")
    ticker = yf.Ticker(pair)

    for i in range(20):
        data = ticker.history(period="1d", interval="1m")
        if not data.empty:
            precio_actual = round(data['Close'].iloc[-1], 5)
        else:
            # Simular variación si no hay cambio
            precio_actual = precios[-1] + np.random.uniform(-0.0005, 0.0005) if precios else 1.1900
            precio_actual = round(precio_actual, 5)

        features = [[precio_actual, precio_actual, precio_actual, precio_actual, 1000]]
        prob = model.predict_proba(features)[0][1]

        confianza_signal = np.random.randint(40, 90)
        precios.append(precio_actual)
        confianzas.append(confianza_signal)

        # Lógica de compra/venta con PnL
        if not posicion_abierta:
            if prob >= probabilidad_minima or confianza_signal >= confianza_minima:
                posicion_abierta = True
                precio_entrada = precio_actual
                signales.append("COMPRA")
                log_event(f"Compra abierta en {precio_actual} (Confianza: {confianza_signal}%, Prob: {prob:.2f})")
            else:
                signales.append("CANCELADO")
                log_event(f"Trade cancelado en {precio_actual} (Confianza: {confianza_signal}%, Prob: {prob:.2f})")
        else:
            if prob < probabilidad_minima and confianza_signal < confianza_minima:
                posicion_abierta = False
                pnl = round((precio_actual - precio_entrada) * (volumen / precio_entrada), 2)
                ganancia_total += pnl
                pnl_por_trade.append(pnl)
                trades_ejecutados += 1
                signales.append("VENTA")
                log_event(f"Venta cerrada en {precio_actual} | PnL: {pnl} (Confianza: {confianza_signal}%, Prob: {prob:.2f})")
            else:
                signales.append("MANTENER")

        # Actualizar gráfico
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=precios, mode='lines+markers', name='Precio'))
        for idx, signal in enumerate(signales):
            color = 'green' if signal == "COMPRA" else ('red' if signal == "VENTA" else 'gray')
            fig.add_trace(go.Scatter(x=[idx], y=[precios[idx]], mode='markers', marker=dict(color=color, size=10), name=signal))
        fig.update_layout(title="Evolución del Precio y Señales", xaxis_title="Ciclo", yaxis_title="Precio")
        placeholder_chart.plotly_chart(fig, use_container_width=True)

        # Actualizar métricas
        pnl_promedio = round(sum(pnl_por_trade)/len(pnl_por_trade), 2) if pnl_por_trade else 0.0
        with placeholder_metrics.container():
            colA, colB, colC, colD = st.columns(4)
            colA.metric("Trades Ejecutados", trades_ejecutados)
            colB.metric("Ganancia Total ($)", f"{ganancia_total:.2f}")
            colC.metric("PnL Promedio ($)", f"{pnl_promedio:.2f}")
            colD.metric("Confianza Promedio (%)", f"{(sum(confianzas)/len(confianzas)) if confianzas else 0:.2f}")

        time.sleep(2)

if detener:
    st.warning("Bot detenido")
    log_event("Bot detenido")

st.subheader("Últimos eventos")
logs = get_logs()
st.table(logs)

if st.button("Limpiar logs antiguos"):
    clear_logs()
    st.success("Logs eliminados")
