import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go

# Configuración de la página web
st.set_page_config(page_title="Módulo de Integrales Dobles", page_icon="∬", layout="wide")

st.title("∬ Laboratorio de Integrales Dobles: Regiones Generales")
st.markdown("Herramienta pedagógica para analizar regiones de integración $R$, determinar límites automáticamente y comparar el **Caso I (Barrido Vertical)** con el **Caso II (Barrido Horizontal)**.")

# =========================================================================
# BARRA LATERAL: INGRESO DE DATOS
# =========================================================================
st.sidebar.header("📥 Parámetros de la Integral")

func_str = st.sidebar.text_input("Función f(x, y):", value="x * y")
eq1_str = st.sidebar.text_input("Curva superior/derecha g2(x) o h2(y):", value="x")
eq2_str = st.sidebar.text_input("Curva inferior/izquierda g1(x) o h1(y):", value="x**2 / 2")

st.sidebar.divider()
st.sidebar.markdown("**Instrucciones:** \nIngresa las curvas que delimitan la región. El sistema calculará analíticamente las intersecciones para establecer los límites numéricos de integración.")

# Botón para ejecutar el análisis
calcular = st.sidebar.button("Calcular Región e Integrales", type="primary")

# =========================================================================
# LÓGICA ANALÍTICA Y RECONSTRUCCIÓN DE REGIONES
# =========================================================================
if calcular:
    try:
        # Símbolos formales algebraicos
        x, y = sp.symbols('x y')
        f_xy = sp.sympify(func_str)
        c1 = sp.sympify(eq1_str)
        c2 = sp.sympify(eq2_str)

        # 1. Hallar puntos de intersección analíticos en el eje X (para el Caso I)
        puntos_interseccion = sp.solve(sp.Eq(c1, c2), x)
        # Filtrar solo soluciones reales numéricas
        puntos_x = sorted([float(pt.evalf()) for pt in puntos_interseccion if pt.is_real])

        if len(puntos_x) < 2:
            st.error("❌ Las curvas ingresadas no se intersectan en al menos dos puntos reales para encerrar una región acotada de manera automática. Prueba con `x` y `x**2 / 2`.")
        else:
            # Tomamos los dos puntos extremos principales de intersección en X
            ax_min, bx_max = puntos_x[0], puntos_x[-1]
            
            # Evaluar cuál es la función superior e inferior en el punto medio del intervalo
            x_medio = (ax_min + bx_max) / 2.0
            y1_medio = float(c1.subs(x, x_medio).evalf())
            y2_medio = float(c2.subs(x, x_medio).evalf())
            
            if y1_medio >= y2_medio:
                g2_sup, g1_inf = c1, c2
            else:
                g2_sup, g1_inf = c2, c1

            # --- CÁLCULO DE LÍMITES PARA EL CASO II (BARRIDO HORIZONTAL) ---
            try:
                despeje_c1 = sp.solve(sp.Eq(y, c1), x)
                despeje_c2 = sp.solve(sp.Eq(y, c2), x)
                
                # Valores extremos en Y correspondientes a las intersecciones
                cy_min = min(float(g1_inf.subs(x, ax_min).evalf()), float(g1_inf.subs(x, bx_max).evalf()))
                dy_max = max(float(g2_sup.subs(x, ax_min).evalf()), float(g2_sup.subs(x, bx_max).evalf()))
                
                y_medio_c2 = (cy_min + dy_max) / 2.0
                # Seleccionar las ramas adecuadas que estén dentro del rango numérico real
                h_c1_candidates = [h for h in despeje_c1 if h.subs(y, y_medio_c2).is_real]
                h_c2_candidates = [h for h in despeje_c2 if h.subs(y, y_medio_c2).is_real]
                
                # Evaluamos cuál curva está a la derecha (mayor x) en el punto medio de Y
                x_c1_eval = float(h_c1_candidates[0].subs(y, y_medio_c2).evalf()) if h_c1_candidates else x_medio
                x_c2_eval = float(h_c2_candidates[0].subs(y, y_medio_c2).evalf()) if h_c2_candidates else x_medio
                
                if x_c1_eval >= x_c2_eval:
                    h2_der = h_c1_candidates[0] if h_c1_candidates else x
                    h1_izq = h_c2_candidates[0] if h_c2_candidates else x
                else:
                    h2_der = h_c2_candidates[0] if h_c2_candidates else x
                    h1_izq = h_c1_candidates[0] if h_c1_candidates else x
                caso_2_posible = True
            except Exception:
                caso_2_posible = False

            # =========================================================================
            # PLANTEAMIENTO Y RESOLUCIÓN DE INTEGRALES (RENDERIZADO LATEX)
            # =========================================================================
            st.header("1. Planteamiento Matemático de las Integrales Iteradas")
            col_eq1, col_eq2 = st.columns(2)
            
            with col_eq1:
                st.subheader("Caso I: Región Tipo I (Barrido Vertical)")
                st.markdown("La región está acotada por valores constantes en $x$, mientras que $y$ varía entre dos funciones de $x$.")
                
                # Resolución paso a paso con SymPy
                integral_interna_y = sp.integrate(f_xy, (y, g1_inf, g2_sup))
                resultado_final_y = sp.integrate(integral_interna_y, (x, ax_min, bx_max))
                
                st.latex(rf"D = \left\{{ (x, y) \mid {ax_min:.2f} \le x \le {bx_max:.2f} \land {sp.latex(g1_inf)} \le y \le {sp.latex(g2_sup)} \right\}}")
                st.latex(rf"I = \int_{{{ax_min:.2f}}}^{{{bx_max:.2f}}} \left( \int_{{{sp.latex(g1_inf)}}}^{{{sp.latex(g2_sup)}}} {sp.latex(f_xy)} \, dy \right) dx = {float(resultado_final_y.evalf()):.4f}")

            with col_eq2:
                st.subheader("Caso II: Región Tipo II (Barrido Horizontal)")
                if caso_2_posible:
                    st.markdown("La región está acotada por valores constantes en $y$, mientras que $x$ varía entre dos funciones de $y$.")
                    
                    integral_interna_x = sp.integrate(f_xy, (x, h1_izq, h2_der))
                    resultado_final_x = sp.integrate(integral_interna_x, (y, cy_min, dy_max))
                    
                    st.latex(rf"D = \left\{{ (x, y) \mid {cy_min:.2f} \le y \le {dy_max:.2f} \land {sp.latex(h1_izq)} \le x \le {sp.latex(h2_der)} \right\}}")
                    st.latex(rf"I = \int_{{{cy_min:.2f}}}^{{{dy_max:.2f}}} \left( \int_{{{sp.latex(h1_izq)}}}^{{{sp.latex(h2_der)}}} {sp.latex(f_xy)} \, dx \right) dy = {float(resultado_final_x.evalf()):.4f}")
                else:
                    st.warning("⚠️ No se pudo realizar el despeje analítico unívoco para el Caso II automáticamente con estas curvas.")

            st.divider()

            # =========================================================================
            # GENERACIÓN DE GRÁFICOS CON INDICADORES DE BARRIDO (COMPATIBILIDAD FIJADA)
            # =========================================================================
            st.header("2. Visualización Pedagógica del Sentido de Barrido")
            col_g1, col_g2 = st.columns(2)

            # Preparación de vectores de muestreo geométrico
            x_vals = np.linspace(ax_min - 0.5, bx_max + 0.5, 200)
            x_region = np.linspace(ax_min, bx_max, 100)
            
            y_sup_vals = [float(g2_sup.subs(x, val).evalf()) for val in x_vals]
            y_inf_vals = [float(g1_inf.subs(x, val).evalf()) for val in x_vals]
            
            y_sup_reg = [float(g2_sup.subs(x, val).evalf()) for val in x_region]
            y_inf_reg = [float(g1_inf.subs(x, val).evalf()) for val in x_region]

            # CONSOLIDACIÓN SEPARADA (Solución al SyntaxError anterior)
            coordenadas_x_region = np.concatenate([x_region, x_region[::-1]])
            coordenadas_y_region = np.concatenate([y_sup_reg, y_inf_reg[::-1]])

            # --- GRÁFICO CASO I: BARRIDO VERTICAL ---
            with col_g1:
                st.subheader("Visualización del Caso I (Flechas Verticales)")
                fig1 = go.Figure()

                # Dibujar curvas fronteras límites
                fig1.add_trace(go.Scatter(x=x_vals, y=y_sup_vals, mode='lines', line=dict(color='#2c3e50', width=2), name=f"y = {g2_sup}"))
                fig1.add_trace(go.Scatter(x=x_vals, y=y_inf_vals, mode='lines', line=dict(color='#2c3e50', width=2, dash='dash'), name=f"y = {g1_inf}"))
                
                # Sombrear la región acotada
                fig1.add_trace(go.Scatter(
                    x=coordenadas_x_region,
                    y=coordenadas_y_region,
                    fill='toself', fillcolor='rgba(52, 152, 219, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'), name='Región R', hoverinfo='none'
                ))

                # Agregar los vectores indicadores del barrido vertical (dy)
                pasos_x = np.linspace(ax_min + (bx_max-ax_min)*0.2, ax_min + (bx_max-ax_min)*0.8, 4)
                for px in pasos_x:
                    y_i = float(g1_inf.subs(x, px).evalf())
                    y_s = float(g2_sup.subs(x, px).evalf())
                    fig1.add_trace(go.Scatter(
                        x=[px, px], y=[y_i, y_s], mode='lines+markers',
                        marker=dict(symbol=['circle', 'arrow-up'], size=[4, 10], color='#e74c3c'),
                        line=dict(color='#e74c3c', width=2.5), showlegend=False, hoverinfo='none'
                    ))

                fig1.update_layout(xaxis_title="Eje X", yaxis_title="Eje Y", template="plotly_white", height=450, showlegend=False)
                fig1.update_xaxes(zeroline=True, zerolinewidth=1.5, zerolinecolor='black', gridcolor='LightGray')
                fig1.update_yaxes(zeroline=True, zerolinewidth=1.5, zerolinecolor='black', gridcolor='LightGray')
                st.plotly_chart(fig1, use_container_width=True)

            # --- GRÁFICO CASO II: BARRIDO HORIZONTAL ---
            with col_g2:
                st.subheader("Visualización del Caso II (Flechas Horizontales)")
                if caso_2_posible:
                    fig2 = go.Figure()

                    # Dibujar curvas fronteras límites
                    fig2.add_trace(go.Scatter(x=x_vals, y=y_sup_vals, mode='lines', line=dict(color='#2c3e50', width=2)))
                    fig2.add_trace(go.Scatter(x=x_vals, y=y_inf_vals, mode='lines', line=dict(color='#2c3e50', width=2, dash='dash')))
                    
                    # Sombrear la región acotada
                    fig2.add_trace(go.Scatter(
                        x=coordenadas_x_region, y=coordenadas_y_region,
                        fill='toself', fillcolor='rgba(46, 204, 113, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'), name='Región R', hoverinfo='none'
                    ))

                    # Agregar los vectores indicadores del barrido horizontal (dx)
                    pasos_y = np.linspace(cy_min + (dy_max-cy_min)*0.2, cy_min + (dy_max-cy_min)*0.8, 4)
                    for py in pasos_y:
                        try:
                            x_izq = float(h1_izq.subs(y, py).evalf())
                            x_der = float(h2_der.subs(y, py).evalf())
                            fig2.add_trace(go.Scatter(
                                x=[x_izq, x_der], y=[py, py], mode='lines+markers',
                                marker=dict(symbol=['circle', 'arrow-right'], size=[4, 10], color='#e67e22'),
                                line=dict(color='#e67e22', width=2.5), showlegend=False, hoverinfo='none'
                            ))
                        except Exception:
                            pass

                    fig2.update_layout(xaxis_title="Eje X", yaxis_title="Eje Y", template="plotly_white", height=450, showlegend=False)
                    fig2.update_xaxes(zeroline=True, zerolinewidth=1.5, zerolinecolor='black', gridcolor='LightGray')
                    fig2.update_yaxes(zeroline=True, zerolinewidth=1.5, zerolinecolor='black', gridcolor='LightGray')
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("El gráfico del Caso II no se puede mostrar si no se definen expresiones algebraicas invertibles respecto a 'y'.")

    except Exception as e:
        st.error(f"Sintaxis o error matemático no controlado: {e}. Asegúrate de usar operadores válidos como `*` y `**`.")
