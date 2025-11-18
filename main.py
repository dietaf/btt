
import streamlit as st
import sqlite3
import time
from datetime import datetime

# ===================== CONFIGURACIÓN DE BASE DE DATOS =====================
DB_NAME = "trading_logs.db"

# Crear tabla si no existe
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

# ===================== INTERFAZ STREAMLIT =====================
st.set_page_config(page_title="Trading Bot", layout="wide")
st.title("🤖 Bot de Trading - TrendShift")

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

# ===================== LÓGICA DEL BOT =====================
if iniciar:
    st.success("Bot iniciado")
    log_event("Bot iniciado con estrategia {} en {}".format(strategy, pair))
    for i in range(5):  # Simulación de 5 ciclos
        precio_actual = round(1.1900 + (i * 0.0005), 5)
        confianza_signal = 58  # Simulación de confianza
        log_event(f"Precio actual: {precio_actual}")
        log_event(f"Señal detectada: COMPRA (Confianza: {confianza_signal}%)")

        if confianza_signal >= confianza_minima:
            log_event("✅ Trade ejecutado (Confianza suficiente)")
        else:
            log_event("❌ Trade cancelado (Confianza muy baja)")

        time.sleep(1)

if detener:
    st.warning("Bot detenido")
    log_event("Bot detenido")

# ===================== MOSTRAR LOGS =====================
st.subheader("Últimos 100 eventos guardados en SQLite")
logs = get_logs()
for ts, msg in logs:
    st.text(f"[{ts}] {msg}")

if st.button("Limpiar logs antiguos"):
    clear_logs()
    st.success("Logs eliminados")
