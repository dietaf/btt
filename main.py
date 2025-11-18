
import streamlit as st
import sqlite3
import time
import os
from datetime import datetime
import plotly.graph_objects as go

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

# ===================== FUNCIONES =====================
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
st.title("📊 Dashboard Profesional - Bot de Trading")

# Panel de configuración
st.sidebar.header("Configuración")
strategy = st.sidebar.selectbox("Estrategia", ["TrendShift", "MeanReversion", "Breakout"])
pair = st.sidebar.selectbox("Par", ["EUR/USD", "GBP/USD", "BTC/USD"])
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m"])

st.sidebar.subheader("Parámetros Iniciales")
volumen = st.sidebar.number_input("Volumen ($)", min_value=1000, max_value=100000, value=10000)
risk = st.sidebar.slider("Riesgo Mínimo (%)", 0, 100, 5)
confianza_minima = st.sidebar.slider("Confianza Mínima (%)", 0, 100, 10)

# Botones
col1, col2 = st.sidebar.columns(2)
iniciar = col1.button("INICIAR")
detener = col2.button("DETENER")

# ===================== SIMULACIÓN DE DATOS =====================
precios = []
signales = []
confianzas = []
trades_ejecutados = 0
ganancia_total = 0.0

if iniciar:
    st.success("Bot iniciado")
    log_event(f"Bot iniciado con estrategia {strategy} en {pair}")
    for i in range(10):
        precio_actual = round(1.1900 + (i * 0.0005), 5)
        confianza_signal = 50 + i  # Simulación confianza creciente
        precios.append(precio_actual)
        confianzas.append(confianza_signal)
        if confianza_signal >= confianza_minima:
            signales.append("COMPRA")
            trades_ejecutados += 1
            ganancia_total += round((precio_actual - 1.1900) * 10000, 2)  # Simulación PnL
            log_event(f"Trade ejecutado en {precio_actual} (Confianza: {confianza_signal}%)")
        else:
            signales.append("CANCELADO")
            log_event(f"Trade cancelado en {precio_actual} (Confianza: {confianza_signal}%)")
        time.sleep(0.5)

if detener:
    st.warning("Bot detenido")
    log_event("Bot detenido")

# ===================== DASHBOARD =====================
st.subheader("Gráfico en tiempo real con señales")
fig = go.Figure()
fig.add_trace(go.Scatter(y=precios, mode='lines+markers', name='Precio'))

# Añadir señales de compra
for idx, signal in enumerate(signales):
    if signal == "COMPRA":
        fig.add_trace(go.Scatter(x=[idx], y=[precios[idx]], mode='markers', marker=dict(color='green', size=10), name='Compra'))
    else:
        fig.add_trace(go.Scatter(x=[idx], y=[precios[idx]], mode='markers', marker=dict(color='red', size=10), name='Cancelado'))

fig.update_layout(title="Evolución del Precio y Señales", xaxis_title="Ciclo", yaxis_title="Precio")
st.plotly_chart(fig, use_container_width=True)

# ===================== MÉTRICAS CLAVE =====================
st.subheader("Métricas del Bot")
colA, colB, colC = st.columns(3)
colA.metric("Trades Ejecutados", trades_ejecutados)
colB.metric("Ganancia Total ($)", f"{ganancia_total:.2f}")
colC.metric("Confianza Promedio (%)", f"{(sum(confianzas)/len(confianzas)) if confianzas else 0:.2f}")

# Mostrar tabla de logs
st.subheader("Últimos eventos")
logs = get_logs()
st.table(logs)

if st.button("Limpiar logs antiguos"):
    clear_logs()
    st.success("Logs eliminados")
