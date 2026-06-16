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
eqA_str = st.sidebar.text_input("Curva Frontera A (puedes usar x o y):", value="y + 1")
eqB_str = st.sidebar.text_input("Curva Frontera B (puedes usar x o y):", value="(y**2 - 6) / 2")

st.sidebar.divider()
st.sidebar.markdown(
    "**Enfoque didáctico multieje:**\n"
    "Tus estudiantes pueden ingresar funciones tanto en términos de $x$ ($y=f(x)$) como en términos de $y$ ($x=g(y)$). "
    "El motor detectará automáticamente el plano de intersección acotado."
)

calcular = st.sidebar.button("Calcular Región e Integrales", type="primary")

# =========================================================================
# LÓGICA ANALÍTICA MULTIEJE Y ADAPTATIVA
# =========================================================================
if calcular:
    try:
        x, y = sp.symbols('x y')
        f_xy = sp.sympify(func_str)
        
        # Leemos las expresiones crudas ingresadas por el usuario
        exprA = sp.sympify(eqA_str)
        exprB = sp.sympify(eqB_str)

        # VARIABLE DE CONTROL DE INTERSECCIÓN
        eje_interseccion = None
        puntos_num = []

        # INTENTO 1: ¿Se intersectan de forma natural en el eje X? (Asumiendo entradas tipo y = f(x))
        try:
            sol_x = sp.solve(sp.Eq(exprA, exprB), x)
            puntos_num = sorted([float(pt.evalf()) for pt in sol_x if pt.is_real])
            if len(puntos_num) >= 2:
                eje_interseccion = 'X'
        except Exception:
            pass

        # INTENTO 2: Si falló el eje X, ¿se intersectan en el eje Y? (Entradas tipo x = g(y))
        if eje_interseccion is None:
            try:
                sol_y = sp.solve(sp.Eq(exprA, exprB), y)
                puntos_num = sorted([float(pt.evalf()) for pt in sol_y if pt.is_real])
                if len(puntos_num) >= 2:
                    eje_interseccion = 'Y'
            except Exception:
                st.error("❌ Las curvas ingresadas no intersectan de forma evidente en el plano real.")

        if eje_interseccion is not None and len(puntos_num) >= 2:
            
            # -----------------------------------------------------------------
            # ESCENARIO A: INTERSECCIÓN ORIGINAL DETECTADA EN EL EJE X
            # -----------------------------------------------------------------
            if eje_interseccion == 'X':
                ax_min, bx_max = puntos_num[0], puntos_num[-1]
                x_medio = (ax_min + bx_max) / 2.0
                
                # Identificar Techo y Piso
                yA_med = float(exprA.subs(x, x_medio).evalf())
                yB_med = float(exprB.subs(x, x_medio).evalf())
                g2_sup, g1_inf = (exprA, exprB) if yA_med >= yB_med else (exprB, exprA)
                
                # Reconstruir límites para el Caso II (Barrido Horizontal)
                y_extremos = [float(g1_inf.subs(x, ax_min).evalf()), float(g1_inf.subs(x, bx_max).evalf()),
                              float(g2_sup.subs(x, ax_min).evalf()), float(g2_sup.subs(x, bx_max).evalf())]
                cy_min, dy_max = min(y_extremos), max(y_extremos)
                
                caso_2_posible = False
                try:
                    despeje_A = sp.solve(sp.Eq(y, exprA), x)
                    despeje_B = sp.solve(sp.Eq(y, exprB), x)
                    y_medio_c2 = (cy_min + dy_max) / 2.0
                    
                    valid_xA = [(float(s.subs(y, y_medio_c2).evalf()), s) for s in despeje_A if s.subs(y, y_medio_c2).is_real]
                    valid_xB = [(float(s.subs(y, y_medio_c2).evalf()), s) for s in despeje_B if s.subs(y, y_medio_c2).is_real]
                    
                    if valid_xA and valid_xB:
                        h2_der, h1_izq = (valid_xA[0][1], valid_xB[0][1]) if valid_xA[0][0] >= valid_xB[0][0] else (valid_xB[0][1], valid_xA[0][1])
                        caso_2_posible = True
                except Exception:
                    caso_2_posible = False

            # -----------------------------------------------------------------
            # ESCENARIO B: INTERSECCIÓN ORIGINAL DETECTADA EN EL EJE Y (Tu caso actual)
            # -----------------------------------------------------------------
            else:
                cy_min, dy_max = puntos_num[0], puntos_num[-1]
                y_medio_c2 = (cy_min + dy_max) / 2.0
                
                # Identificar Derecha e Izquierda directamente
                xA_med = float(exprA.subs(y, y_medio_c2).evalf())
                xB_med = float(exprB.subs(y, y_medio_c2).evalf())
                h2_der, h1_izq = (exprA, exprB) if xA_med >= xB_med else (exprB, exprA)
                
                # Reconstruir límites para el Caso I (Barrido Vertical)
                x_extremos = [float(h1_izq.subs(y, cy_min).evalf()), float(h1_izq.subs(y, dy_max).evalf()),
                              float(h2_der.subs(y, cy_min).evalf()), float(h2_der.subs(y, dy_max).evalf())]
                ax_min, bx_max = min(x_extremos), max(x_extremos)
                
                caso_2_posible = True # El caso horizontal es el nativo aquí
                
                # Intentamos despejar y en función de x para las flechas verticales del Caso I
                try:
                    despeje_yA = sp.solve(sp.Eq(x, exprA), y)
                    despeje_yB = sp.solve(sp.Eq(x, exprB), y)
                    x_medio = (ax_min + bx_max) / 2.0
                    
                    valid_yA = [(float(s.subs(x, x_medio).evalf()), s) for s in despeje_yA if s.subs(x, x_medio).is_real]
                    valid_yB = [(float(s.subs(x, x_medio).evalf()), s) for s in despeje_yB if s.subs(x, x_medio).is_real]
                    
                    if valid_yA and valid_yB:
                        g2_sup, g1_inf = (valid_yA[0][1], valid_yB[0][1]) if valid_yA[0][0] >= valid_yB[0][0] else (valid_yB[0][1], valid_yA[0][1])
                    else:
                        # Fallback seguro aproximado para el renderizado gráfico si no se puede despejar de forma única
                        g2_sup, g1_inf = sp.solve(sp.Eq(x, h2_der), y)[0], sp.solve(sp.Eq(x, h1_izq), y)[0]
                except Exception:
                    g2_sup, g1_inf = y, y

            # =========================================================================
            # PLANTEAMIENTO MATEMÁTICO CON PASO A PASO DIDÁCTICO
            # =========================================================================
            st.header("1. Desarrollo Analítico Paso a Paso de las Integrales Iteradas")
            col_eq1, col_eq2 = st.columns(2)
            
            with col_eq1:
                st.subheader("Caso I: Región Tipo I (Barrido Vertical)")
                st.markdown("**A) Planteamiento formal de la Integral:**")
                st.latex(rf"I = \int_{{{ax_min:.2f}}}^{{{bx_max:.2f}}} \left( \int_{{{sp.latex(g1_inf)}}}^{{{sp.latex(g2_sup)}}} {sp.latex(f_xy)} \, dy \right) dx")
                
                antiderivada_y = sp.integrate(f_xy, y)
                st.markdown("**B) Antiderivada interna calculada (con respecto a $y$):**")
                st.latex(rf"\left. {sp.latex(antiderivada_y)} \right|_{{{sp.latex(g1_inf)}}}^{{{sp.latex(g2_sup)}}}")
                
                integral_externa_reemplazada = sp.integrate(f_xy, (y, g1_inf, g2_sup))
                st.markdown("**C) Expresión resultante para la integral exterior:**")
                st.latex(rf"I = \int_{{{ax_min:.2f}}}^{{{bx_max:.2f}}} \left( {sp.latex(integral_externa_reemplazada)} \right) dx")
                
                resultado_final_y = sp.integrate(integral_externa_reemplazada, (x, ax_min, bx_max))
                st.markdown(f"**Resultado Final:** $\\mathbf{{I = {float(resultado_final_y.evalf()):.4f}}}$")

            with col_eq2:
                st.subheader("Caso II: Región Tipo II (Barrido Horizontal)")
                if caso_2_posible:
                    st.markdown("**A) Planteamiento formal de la Integral:**")
                    st.latex(rf"I = \int_{{{cy_min:.2f}}}^{{{dy_max:.2f}}} \left( \int_{{{sp.latex(h1_izq)}}}^{{{sp.latex(h2_der)}}} {sp.latex(f_xy)} \, dx \right) dy")
                    
                    antiderivada_x = sp.integrate(f_xy, x)
                    st.markdown("**B) Antiderivada interna calculada (con respecto a $x$):**")
                    st.latex(rf"\left. {sp.latex(antiderivada_x)} \right|_{{{sp.latex(h1_izq)}}}^{{{sp.latex(h2_der)}}}")
                    
                    integral_externa_reemplazada_x = sp.integrate(f_xy, (x, h1_izq, h2_der))
                    st.markdown("**C) Expresión resultante para la integral exterior:**")
                    st.latex(rf"I = \int_{{{cy_min:.2f}}}^{{{dy_max:.2f}}} \left( {sp.latex(integral_externa_reemplazada_x)} \right) dy")
                    
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

            # Muestreos numéricos adaptados al contexto real acotado
            y_mesh = np.linspace(cy_min, dy_max, 200)
            
            # Obtener puntos numéricos de las fronteras izquierda y derecha para graficar la superficie limpia
            x_izq_vals = [float(h1_izq.subs(y, val).evalf()) for val in y_mesh]
            x_der_vals = [float(h2_der.subs(y, val).evalf()) for val in y_mesh]
            
            # Unión perimetral para el sombreado relleno de Plotly
            coordenadas_x_region = np.concatenate([x_izq_vals, x_der_vals[::-1]])
            coordenadas_y_region = np.concatenate([y_mesh, y_mesh[::-1]])

            # Extender el lienzo un poco más allá de los límites para que se aprecie la intersección
            y_canvas = np.linspace(cy_min - 1.0, dy_max + 1.0, 250)
            x_izq_canvas = [float(h1_izq.subs(y, val).evalf()) for val in y_canvas]
            x_der_canvas = [float(h2_der.subs(y, val).evalf()) for val in y_canvas]

            # --- GRÁFICO CASO I: BARRIDO VERTICAL ---
            with col_g1:
                st.subheader("Visualización del Caso I (Flechas Verticales)")
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=x_izq_canvas, y=y_canvas, mode='lines', line=dict(color='#2c3e50', width=2), name="Curva Frontera"))
                fig1.add_trace(go.Scatter(x=x_der_canvas, y=y_canvas, mode='lines', line=dict(color='#2c3e50', width=2, dash='dash'), name="Curva Frontera"))
                
                fig1.add_trace(go.Scatter(
                    x=coordenadas_x_region, y=coordenadas_y_region,
                    fill='toself', fillcolor='rgba(52, 152, 219, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'), showlegend=False, hoverinfo='none'
                ))

                # Flechas verticales distribuidas
                pasos_x = np.linspace(ax_min + (bx_max-ax_min)*0.2, ax_min + (bx_max-ax_min)*0.8, 4)
                for px in pasos_x:
                    try:
                        # Evaluamos la altura resolviendo numéricamente las fronteras en esa sección de X
                        y_sols = sorted([float(s.subs(x, px).evalf()) for s in sp.solve(sp.Eq(x, h1_izq), y) + sp.solve(sp.Eq(x, h2_der), y) if s.subs(x, px).is_real])
                        if len(y_sols) >= 2:
                            fig1.add_trace(go.Scatter(
                                x=[px, px], y=[y_sols[0], y_sols[-1]], mode='lines+markers',
                                marker=dict(symbol=['circle', 'arrow-up'], size=[4, 10], color='#e74c3c'),
                                line=dict(color='#e74c3c', width=2.5), showlegend=False, hoverinfo='none'
                            ))
                    except Exception:
                        pass

                fig1.update_layout(xaxis_title="Eje X", yaxis_title="Eje Y", template="plotly_white", height=450, showlegend=False)
                st.plotly_chart(fig1, use_container_width=True)

            # --- GRÁFICO CASO II: BARRIDO HORIZONTAL ---
            with col_g2:
                st.subheader("Visualización del Caso II (Flechas Horizontales)")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=x_izq_canvas, y=y_canvas, mode='lines', line=dict(color='#2c3e50', width=2)))
                fig2.add_trace(go.Scatter(x=x_der_canvas, y=y_canvas, mode='lines', line=dict(color='#2c3e50', width=2, dash='dash')))
                
                fig2.add_trace(go.Scatter(
                    x=coordenadas_x_region, y=coordenadas_y_region,
                    fill='toself', fillcolor='rgba(46, 204, 113, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'), showlegend=False, hoverinfo='none'
                ))

                # Flechas horizontales distribuidas (Ideales para este caso)
                pasos_y = np.linspace(cy_min + (dy_max-cy_min)*0.2, cy_min + (dy_max-cy_min)*0.8, 4)
                for py in pasos_y:
                    x_izq_f = float(h1_izq.subs(y, py).evalf())
                    x_der_f = float(h2_der.subs(y, py).evalf())
                    fig2.add_trace(go.Scatter(
                        x=[x_izq_f, x_der_f], y=[py, py], mode='lines+markers',
                        marker=dict(symbol=['circle', 'arrow-right'], size=[4, 10], color='#e67e22'),
                        line=dict(color='#e67e22', width=2.5), showlegend=False, hoverinfo='none'
                    ))

                fig2.update_layout(xaxis_title="Eje X", yaxis_title="Eje Y", template="plotly_white", height=450, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Error general de análisis: {e}. Asegúrate de escribir expresiones compatibles con SymPy.")
