
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
st.title("📈 Dashboard Profesional - Bot de Trading con ML Optimizado")

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

# ===================== ENTRENAR MODELO ML CON OPTIMIZACIÓN =====================
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
        'n_estimators': [50],  # reducido para optimización rápida
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

if iniciar:
    st.success("Bot iniciado")
    log_event(f"Bot iniciado con estrategia {strategy} en {pair}")
    ticker = yf.Ticker(pair)

    for i in range(20):
        data = ticker.history(period="1d", interval="1m")
        if not data.empty:
            precio_actual = round(data['Close'].iloc[-1], 5)
            features = [[data['Open'].iloc[-1], data['High'].iloc[-1], data['Low'].iloc[-1], data['Close'].iloc[-1], data['Volume'].iloc[-1]]]
            prob = model.predict_proba(features)[0][1]
        else:
            precio_actual = round(1.1900 + np.random.uniform(-0.0005, 0.0005), 5)
            prob = np.random.uniform(0.4, 0.9)

        confianza_signal = np.random.randint(40, 90)
        precios.append(precio_actual)
        confianzas.append(confianza_signal)

        if prob >= probabilidad_minima or confianza_signal >= confianza_minima:
            signales.append("COMPRA")
            trades_ejecutados += 1
            ganancia_total += round((precio_actual - precios[0]) * 10000, 2)
            log_event(f"Trade ejecutado en {precio_actual} (Confianza: {confianza_signal}%, Prob: {prob:.2f})")
        else:
            signales.append("CANCELADO")
            log_event(f"Trade cancelado en {precio_actual} (Confianza: {confianza_signal}%, Prob: {prob:.2f})")

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=precios, mode='lines+markers', name='Precio'))
        for idx, signal in enumerate(signales):
            color = 'green' if signal == "COMPRA" else 'red'
            fig.add_trace(go.Scatter(x=[idx], y=[precios[idx]], mode='markers', marker=dict(color=color, size=10), name=signal))
        fig.update_layout(title="Evolución del Precio y Señales", xaxis_title="Ciclo", yaxis_title="Precio")
        placeholder_chart.plotly_chart(fig, use_container_width=True)

        with placeholder_metrics.container():
            colA, colB, colC = st.columns(3)
            colA.metric("Trades Ejecutados", trades_ejecutados)
            colB.metric("Ganancia Total ($)", f"{ganancia_total:.2f}")
            colC.metric("Confianza Promedio (%)", f"{(sum(confianzas)/len(confianzas)) if confianzas else 0:.2f}")

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
