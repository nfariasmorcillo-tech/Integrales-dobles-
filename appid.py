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
eqA_str = st.sidebar.text_input("Curva Frontera A: y = f(x) o x = g(y)", value="x")
eqB_str = st.sidebar.text_input("Curva Frontera B: y = f(x) o x = g(y)", value="x**2 / 2")

st.sidebar.divider()
st.sidebar.markdown(
    "**Enfoque didáctico:**\n"
    "Introduce las dos funciones en cualquier orden. El sistema identificará automáticamente "
    "las intersecciones y deducirá los límites superiores e inferiores para ambos casos."
)

# Botón para ejecutar el análisis
calcular = st.sidebar.button("Calcular Región e Integrales", type="primary")

# =========================================================================
# LÓGICA ANALÍTICA AUTOMATIZADA
# =========================================================================
if calcular:
    try:
        # Símbolos formales algebraicos
        x, y = sp.symbols('x y')
        f_xy = sp.sympify(func_str)
        cA = sp.sympify(eqA_str)
        cB = sp.sympify(eqB_str)

        # 1. Hallar puntos de intersección analíticos en el eje X
        puntos_interseccion = sp.solve(sp.Eq(cA, cB), x)
        puntos_x = sorted([float(pt.evalf()) for pt in puntos_interseccion if pt.is_real])

        if len(puntos_x) < 2:
            st.error("❌ Las curvas ingresadas no encierran una región acotada de forma automática (se necesitan al menos 2 puntos de intersección reales). Revisa las expresiones.")
        else:
            # Límites definitivos en X (Caso I)
            ax_min, bx_max = puntos_x[0], puntos_x[-1]
            
            # DETERMINACIÓN DINÁMICA CASO I: ¿Cuál es techo y cuál es piso?
            x_medio = (ax_min + bx_max) / 2.0
            yA_medio = float(cA.subs(x, x_medio).evalf())
            yB_medio = float(cB.subs(x, x_medio).evalf())
            
            if yA_medio >= yB_medio:
                g2_sup, g1_inf = cA, cB
            else:
                g2_sup, g1_inf = cB, cA

            # --- PREPARACIÓN AUTOMÁTICA PARA EL CASO II (BARRIDO HORIZONTAL) ---
            y_extremos = [float(g1_inf.subs(x, ax_min).evalf()), float(g1_inf.subs(x, bx_max).evalf()),
                          float(g2_sup.subs(x, ax_min).evalf()), float(g2_sup.subs(x, bx_max).evalf())]
            cy_min, dy_max = min(y_extremos), max(y_extremos)
            
            caso_2_posible = False
            try:
                # Despejamos x en función de y para ambas curvas de entrada
                despeje_A = sp.solve(sp.Eq(y, cA), x)
                despeje_B = sp.solve(sp.Eq(y, cB), x)
                
                y_medio_c2 = (cy_min + dy_max) / 2.0
                
                # --- FILTRO GEOMÉTRICO ROBUSTO DE RAMAS ---
                # Evaluamos los valores reales de x en el centro vertical de la región para elegir la rama correcta
                valid_xA = []
                for sol in despeje_A:
                    val_eval = sol.subs(y, y_medio_c2)
                    if val_eval.is_real and (ax_min <= float(val_eval.evalf()) <= bx_max):
                        valid_xA.append((float(val_eval.evalf()), sol))
                        
                valid_xB = []
                for sol in despeje_B:
                    val_eval = sol.subs(y, y_medio_c2)
                    if val_eval.is_real and (ax_min <= float(val_eval.evalf()) <= bx_max):
                        valid_xB.append((float(val_eval.evalf()), sol))

                if valid_xA and valid_xB:
                    # Ordenamos por su posición real en el eje X para asegurar consistencia de signos
                    x_A_real, rama_A_elegida = valid_xA[0]
                    x_B_real, rama_B_elegida = valid_xB[0]
                    
                    if x_A_real >= x_B_real:
                        h2_der, h1_izq = rama_A_elegida, rama_B_elegida
                    else:
                        h2_der, h1_izq = rama_B_elegida, rama_A_elegida
                    caso_2_posible = True
            except Exception:
                caso_2_posible = False

            # =========================================================================
            # PLANTEAMIENTO MATEMÁTICO CON PASO A PASO DIDÁCTICO
            # =========================================================================
            st.header("1. Desarrollo Analítico Paso a Paso de las Integrales Iteradas")
            col_eq1, col_eq2 = st.columns(2)
            
            with col_eq1:
                st.subheader("Caso I: Región Tipo I (Barrido Vertical)")
                
                # Paso 1: Planteamiento
                st.markdown("**A) Planteamiento formal de la Integral:**")
                st.latex(rf"I = \int_{{{ax_min:.2f}}}^{{{bx_max:.2f}}} \left( \int_{{{sp.latex(g1_inf)}}}^{{{sp.latex(g2_sup)}}} {sp.latex(f_xy)} \, dy \right) dx")
                
                # Paso 2: Antiderivada Interna sin evaluar
                antiderivada_y = sp.integrate(f_xy, y)
                st.markdown("**B) Antiderivada interna calculada (con respecto a $y$):**")
                st.latex(rf"\left. {sp.latex(antiderivada_y)} \right|_{{{sp.latex(g1_inf)}}}^{{{sp.latex(g2_sup)}}}")
                
                # Paso 3: Sustitución de límites en la interna para armar la externa
                integral_externa_reemplazada = sp.integrate(f_xy, (y, g1_inf, g2_sup))
                st.markdown("**C) Expresión resultante para la integral exterior:**")
                st.latex(rf"I = \int_{{{ax_min:.2f}}}^{{{bx_max:.2f}}} \left( {sp.latex(integral_externa_reemplazada)} \right) dx")
                
                # Resultado final numérico
                resultado_final_y = sp.integrate(integral_externa_reemplazada, (x, ax_min, bx_max))
                st.markdown(f"**Resultado Final:** $\\mathbf{{I = {float(resultado_final_y.evalf()):.4f}}}$")

            with col_eq2:
                st.subheader("Caso II: Región Tipo II (Barrido Horizontal)")
                if caso_2_posible:
                    # Paso 1: Planteamiento
                    st.markdown("**A) Planteamiento formal de la Integral:**")
                    st.latex(rf"I = \int_{{{cy_min:.2f}}}^{{{dy_max:.2f}}} \left( \int_{{{sp.latex(h1_izq)}}}^{{{sp.latex(h2_der)}}} {sp.latex(f_xy)} \, dx \right) dy")
                    
                    # Paso 2: Antiderivada Interna sin evaluar
                    antiderivada_x = sp.integrate(f_xy, x)
                    st.markdown("**B) Antiderivada interna calculada (con respecto a $x$):**")
                    st.latex(rf"\left. {sp.latex(antiderivada_x)} \right|_{{{sp.latex(h1_izq)}}}^{{{sp.latex(h2_der)}}}")
                    
                    # Paso 3: Sustitución de límites en la interna para armar la externa
                    integral_externa_reemplazada_x = sp.integrate(f_xy, (x, h1_izq, h2_der))
                    st.markdown("**C) Expresión resultante para la integral exterior:**")
                    st.latex(rf"I = \int_{{{cy_min:.2f}}}^{{{dy_max:.2f}}} \left( {sp.latex(integral_externa_reemplazada_x)} \right) dy")
                    
                    # Resultado final numérico
                    resultado_final_x = sp.integrate(integral_externa_reemplazada_x, (y, cy_min, dy_max))
                    st.markdown(f"**Resultado Final:** $\\mathbf{{I = {float(resultado_final_x.evalf()):.4f}}}$")
                else:
                    st.warning("⚠️ No se pudo automatizar el despeje unívoco de las inversas en el eje Y para este par de curvas.")

            st.divider()

            # =========================================================================
            # GRÁFICOS CON INDICADORES GEOMÉTRICOS DE BARRIDO
            # =========================================================================
            st.header("2. Visualización Pedagógica del Sentido de Barrido")
            col_g1, col_g2 = st.columns(2)

            # Muestreos numéricos estables
            x_vals = np.linspace(ax_min - 0.5, bx_max + 0.5, 200)
            x_region = np.linspace(ax_min, bx_max, 100)
            
            y_sup_vals = [float(g2_sup.subs(x, val).evalf()) for val in x_vals]
            y_inf_vals = [float(g1_inf.subs(x, val).evalf()) for val in x_vals]
            
            y_sup_reg = [float(g2_sup.subs(x, val).evalf()) for val in x_region]
            y_inf_reg = [float(g1_inf.subs(x, val).evalf()) for val in x_region]

            coordenadas_x_region = np.concatenate([x_region, x_region[::-1]])
            coordenadas_y_region = np.concatenate([y_sup_reg, y_inf_reg[::-1]])

            # --- GRÁFICO CASO I: BARRIDO VERTICAL ---
            with col_g1:
                st.subheader("Visualización del Caso I (Flechas Verticales)")
                fig1 = go.Figure()

                fig1.add_trace(go.Scatter(x=x_vals, y=y_sup_vals, mode='lines', line=dict(color='#2c3e50', width=2.5), name="Techo geométrico"))
                fig1.add_trace(go.Scatter(x=x_vals, y=y_inf_vals, mode='lines', line=dict(color='#34495e', width=2, dash='dash'), name="Piso geométrico"))
                
                fig1.add_trace(go.Scatter(
                    x=coordenadas_x_region, y=coordenadas_y_region,
                    fill='toself', fillcolor='rgba(52, 152, 219, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'), name='Región R', hoverinfo='none'
                ))

                # Flechas indicadoras verticales dinámicas
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

                    fig2.add_trace(go.Scatter(x=x_vals, y=y_sup_vals, mode='lines', line=dict(color='#2c3e50', width=2.5)))
                    fig2.add_trace(go.Scatter(x=x_vals, y=y_inf_vals, mode='lines', line=dict(color='#34495e', width=2, dash='dash')))
                    
                    fig2.add_trace(go.Scatter(
                        x=coordenadas_x_region, y=coordenadas_y_region,
                        fill='toself', fillcolor='rgba(46, 204, 113, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'), name='Región R', hoverinfo='none'
                    ))

                    # Flechas indicadoras horizontales dinámicas
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
                    st.info("El gráfico de barrido horizontal requiere curvas cuyas funciones inversas directas existan analíticamente.")

    except Exception as e:
        st.error(f"Error en el procesamiento: {e}. Revisa la coherencia algebraica de los datos introducidos.")
