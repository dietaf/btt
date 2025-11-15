def main():
    # ===================================================================
    # SISTEMA DE AUTENTICACIÓN
    # ===================================================================
    
    # Inicializar estado de autenticación
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Si no está autenticado, mostrar formulario de login
    if not st.session_state.authenticated:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 30px;">
            <h1 style="color: white; margin: 0;">🔐 Bot Trading Profesional</h1>
            <p style="color: white; margin-top: 10px;">Acceso Restringido</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 🔑 Iniciar Sesión")
            
            # CAMBIA ESTA CONTRASEÑA POR LA TUYA
            MASTER_PASSWORD = "Trading2024$"  # ⚠️ CAMBIAR ESTO
            
            password = st.text_input("Contraseña:", type="password", key="password_input")
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("🔓 Acceder", use_container_width=True, type="primary"):
                    if password == MASTER_PASSWORD:
                        st.session_state.authenticated = True
                        st.success("✅ Acceso concedido")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Contraseña incorrecta")
            
            with col_btn2:
                if st.button("❓ Ayuda", use_container_width=True):
                    st.info("""
                    **¿Olvidaste la contraseña?**
                    
                    Edita el archivo `main.py` en GitHub:
                    
                    Línea ~700:
                    ```python
                    MASTER_PASSWORD = "TuNuevaContraseña"
                    ```
                    
                    Guarda y espera 2 minutos para redeploy.
                    """)
            
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; color: #666; font-size: 0.9em;">
                🔒 Protegido por contraseña<br>
                🧠 Machine Learning Activado<br>
                💾 SQLite Database<br>
            </div>
            """, unsafe_allow_html=True)
        
        return  # Detener ejecución si no está autenticado
    
    # ===================================================================
    # APLICACIÓN PRINCIPAL (Solo si está autenticado)
    # ===================================================================
    
    st.title("🧠 Bot Trading Profesional - ML + SQLite")
    st.markdown("### Machine Learning | Auto-Optimización | Backtesting")
    
    if 'bot' not in st.session_state:
        st.session_state.bot = None
        st.session_state.bot_running = False
