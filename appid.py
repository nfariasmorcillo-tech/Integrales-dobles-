import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go

# Configuración de la página web
st.set_page_config(page_title="Módulo de Integrales Dobles", page_icon="∬", layout="wide")

st.title("∬ Laboratorio de Integrales Dobles: Regiones Generales")
st.markdown("Herramienta pedagógica para analizar regiones de integración $R$, determinar límites automáticamente y comparar el **Caso I (Barrido Vertical)** con el **Caso II (Barrido Horizontal)**.")

# =========================================================================
# BARRA LATERAL: INGRESO DE DATOS CANÓNICOS
# =========================================================================
st.sidebar.header("📥 Parámetros de la Integral")

func_str = st.sidebar.text_input("Función f(x, y):", value="x * y")
eqA_str = st.sidebar.text_input("Curva Frontera A (y=... o x=...):", value="y + 1")
eqB_str = st.sidebar.text_input("Curva Frontera B (y=... o x=...):", value="(y**2 - 6) / 2")

st.sidebar.divider()
st.sidebar.markdown(
    "**Sugerencia Didáctica:**\n"
    "El sistema autodetecta las variables y valida algebraicamente las ramas de raíces cuadradas "
    "mediante comprobación cruzada en los puntos medios de la región."
)

calcular = st.sidebar.button("Calcular Región e Integrales", type="primary")

# =========================================================================
# MOTOR ALGEBRAICO ADAPTATIVO CON VALIDACIÓN CRUZADA
# =========================================================================
if calcular:
    try:
        x, y = sp.symbols('x y')
        f_xy = sp.sympify(func_str)
        
        # Parseo flexible: Determinar si el alumno metió funciones respecto a X o respecto a Y
        rawA = sp.sympify(eqA_str)
        rawB = sp.sympify(eqB_str)

        # Normalizamos ambas expresiones a ecuaciones explícitas en términos de la variable dominante
        # Para lograrlo de forma infalible, buscamos los puntos de intersección en ambos ejes de forma paralela
        sols_x = []
        try: sols_x = [float(pt.evalf()) for pt in sp.solve(sp.Eq(rawA, rawB), x) if pt.is_real]
        except: pass

        sols_y = []
        try: sols_y = [float(pt.evalf()) for pt in sp.solve(sp.Eq(rawA, rawB), y) if pt.is_real]
        except: pass

        # Decisión del eje base de anclaje de la región acotada
        if len(sols_y) >= 2:
            # Región definida nativamente de forma horizontal Tipo II
            cy_min, dy_max = min(sols_y), max(sols_y)
            y_medio = (cy_min + dy_max) / 2.0
            
            # Evaluar cuál expresión queda a la derecha (mayor x) y cuál a la izquierda (menor x)
            valA = float(rawA.subs(y, y_medio).evalf())
            valB = float(rawB.subs(y, y_medio).evalf())
            
            h2_der = rawA if valA >= valB else rawB
            h1_izq = rawB if valA >= valB else rawA
            
            # Obtener límites numéricos extremos de X para el Caso I
            ax_min = min(float(h1_izq.subs(y, cy_min).evalf()), float(h1_izq.subs(y, dy_max).evalf()))
            bx_max = max(float(h2_der.subs(y, cy_min).evalf()), float(h2_der.subs(y, dy_max).evalf()))
            
            # --- DESPEJE COMPROBADO Y CRUZADO PARA EL CASO I (Y en función de X) ---
            x_medio = (ax_min + bx_max) / 2.0
            
            # Despejar y de la frontera izquierda y derecha
            ramas_izq = sp.solve(sp.Eq(x, h1_izq), y)
            ramas_der = sp.solve(sp.Eq(x, h2_der), y)
            
            # Juntamos todas las ramas candidatas y seleccionamos de forma cruzada basándonos en la geometría real
            todas_las_ramas = ramas_izq + ramas_der
            ramas_validas = []
            for r in todas_las_ramas:
                val_r = r.subs(x, x_medio)
                if val_r.is_real and (cy_min <= float(val_r.evalf()) <= dy_max):
                    ramas_validas.append((float(val_r.evalf()), r))
            
            ramas_validas = sorted(list(set(ramas_validas)), key=lambda t: t[0])
            
            if len(ramas_validas) >= 2:
                g1_inf = ramas_validas[0][1]
                g2_sup = ramas_validas[-1][1]
            else:
                # Fallback algebraico directo si es lineal
                g1_inf = sp.solve(sp.Eq(x, h2_der), y)[0]
                g2_sup = sp.solve(sp.Eq(x, h1_izq), y)[0]
                if g1_inf.subs(x, x_medio).evalf() > g2_sup.subs(x, x_medio).evalf():
                    g1_inf, g2_sup = g2_sup, g1_inf

        else:
            # Región definida nativamente de forma vertical Tipo I
            ax_min, bx_max = min(sols_x), max(sols_x)
            x_medio = (ax_min + bx_max) / 2.0
            
            valA = float(rawA.subs(x, x_medio).evalf())
            valB = float(rawB.subs(x, x_medio).evalf())
            
            g2_sup = rawA if valA >= valB else rawB
            g1_inf = rawB if valA >= valB else rawA
            
            cy_min = min(float(g1_inf.subs(x, ax_min).evalf()), float(g1_inf.subs(x, bx_max).evalf()))
            dy_max = max(float(g2_sup.subs(x, ax_min).evalf()), float(g2_sup.subs(x, bx_max).evalf()))
            
            # --- DESPEJE COMPROBADO Y CRUZADO PARA EL CASO II (X en función de Y) ---
            y_medio = (cy_min + dy_max) / 2.0
            ramas_inf = sp.solve(sp.Eq(y, g1_inf), x)
            ramas_sup = sp.solve(sp.Eq(y, g2_sup), x)
            
            todas_las_ramas = ramas_inf + ramas_sup
            ramas_validas = []
            for r in todas_las_ramas:
                val_r = r.subs(y, y_medio)
                if val_r.is_real and (ax_min <= float(val_r.evalf()) <= bx_max):
                    ramas_validas.append((float(val_r.evalf()), r))
            
            ramas_validas = sorted(list(set(ramas_validas)), key=lambda t: t[0])
            
            if len(ramas_validas) >= 2:
                h1_izq = ramas_validas[0][1]
                h2_der = ramas_validas[-1][1]
            else:
                h1_izq = sp.solve(sp.Eq(y, g2_sup), x)[0]
                h2_der = sp.solve(sp.Eq(y, g1_inf), x)[0]
                if h1_izq.subs(y, y_medio).evalf() > h2_der.subs(y, y_medio).evalf():
                    h1_izq, h2_der = h2_der, h1_izq

        # =========================================================================
        # RENDERIZADO MATEMÁTICO DESGLOSADO PASO A PASO
        # =========================================================================
        st.header("1. Desarrollo Analítico Paso a Paso de las Integrales Iteradas")
        col_eq1, col_eq2 = st.columns(2)
        
        with col_eq1:
            st.subheader("Caso I: Región Tipo I (Barrido Vertical)")
            st.markdown("**A) Planteamiento formal de la Integral:**")
            st.latex(rf"I = \int_{{{ax_min:.2f}}}^{{{bx_max:.2f}}} \left( \int_{{{sp.latex(g1_inf)}}}^{{{sp.latex(g2_sup)}}} {sp.latex(f_xy)} \, dy \right) dx")
            
            antiderivada_y = sp.integrate(f_xy, y)
            st.markdown("**B) Antiderivada interna calculada (sin evaluar límites):**")
            st.latex(rf"\int {sp.latex(f_xy)} \, dy = {sp.latex(antiderivada_y)}")
            st.markdown("**Evaluación de los límites en la antiderivada interna:**")
            st.latex(rf"\left. {sp.latex(antiderivada_y)} \right|_{{{sp.latex(g1_inf)}}}^{{{sp.latex(g2_sup)}}}")
            
            integral_externa_reemplazada = sp.integrate(f_xy, (y, g1_inf, g2_sup))
            st.markdown("**C) Expresión resultante para la integral exterior (simplificada):**")
            st.latex(rf"I = \int_{{{ax_min:.2f}}}^{{{bx_max:.2f}}} \left( {sp.latex(integral_externa_reemplazada)} \right) dx")
            
            antiderivada_x = sp.integrate(integral_externa_reemplazada, x)
            st.markdown("**D) Antiderivada externa calculada (antes del valor numérico final):**")
            st.latex(rf"\left. {sp.latex(antiderivada_x)} \right|_{{{ax_min:.2f}}}^{{{bx_max:.2f}}}")
            
            resultado_final_y = sp.integrate(integral_externa_reemplazada, (x, ax_min, bx_max))
            st.markdown(f"🎉 **Resultado Numérico Final Case I:** $\\mathbf{{I = {float(resultado_final_y.evalf()):.4f}}}$")

        with col_eq2:
            st.subheader("Caso II: Región Tipo II (Barrido Horizontal)")
            st.markdown("**A) Planteamiento formal de la Integral:**")
            st.latex(rf"I = \int_{{{cy_min:.2f}}}^{{{dy_max:.2f}}} \left( \int_{{{sp.latex(h1_izq)}}}^{{{sp.latex(h2_der)}}} {sp.latex(f_xy)} \, dx \right) dy")
            
            antiderivada_x_indef = sp.integrate(f_xy, x)
            st.markdown("**B) Antiderivada interna calculada (sin evaluar límites):**")
            st.latex(rf"\int {sp.latex(f_xy)} \, dx = {sp.latex(antiderivada_x_indef)}")
            st.markdown("**Evaluación de los límites en la antiderivada interna:**")
            st.latex(rf"\left. {sp.latex(antiderivada_x_indef)} \right|_{{{sp.latex(h1_izq)}}}^{{{sp.latex(h2_der)}}}")
            
            integral_externa_reemplazada_x = sp.integrate(f_xy, (x, h1_izq, h2_der))
            st.markdown("**C) Expresión resultante para la integral exterior (simplificada):**")
            st.latex(rf"I = \int_{{{cy_min:.2f}}}^{{{dy_max:.2f}}} \left( {sp.latex(integral_externa_reemplazada_x)} \right) dy")
            
            antiderivada_y_ext = sp.integrate(integral_externa_reemplazada_x, y)
            st.markdown("**D) Antiderivada externa calculada (antes del valor numérico final):**")
            st.latex(rf"\left. {sp.latex(antiderivada_y_ext)} \right|_{{{cy_min:.2f}}}^{{{dy_max:.2f}}}")
            
            resultado_final_x = sp.integrate(integral_externa_reemplazada_x, (y, cy_min, dy_max))
            st.markdown(f"🎉 **Resultado Numérico Final Case II:** $\\mathbf{{I = {float(resultado_final_x.evalf()):.4f}}}$")

        st.divider()

        # =========================================================================
        # GRÁFICOS INTERACTIVOS PLOTLY CON FILTRO DE MALLA REAL
        # =========================================================================
        st.header("2. Visualización Pedagógica del Sentido de Barrido")
        col_g1, col_g2 = st.columns(2)

        # Muestreo adaptativo limpio para las curvas y el área de la región acotada
        y_mesh = np.linspace(cy_min, dy_max, 200)
        x_izq_vals = [float(h1_izq.subs(y, val).evalf()) for val in y_mesh]
        x_der_vals = [float(h2_der.subs(y, val).evalf()) for val in y_mesh]
        
        coordenadas_x_region = np.concatenate([x_izq_vals, x_der_vals[::-1]])
        coordenadas_y_region = np.concatenate([y_mesh, y_mesh[::-1]])

        y_canvas = np.linspace(cy_min - 1.0, dy_max + 1.0, 250)
        x_izq_canvas = [float(h1_izq.subs(y, val).evalf()) for val in y_canvas]
        x_der_canvas = [float(h2_der.subs(y, val).evalf()) for val in y_canvas]

        # --- FIGURA 1: BARRIDO VERTICAL ---
        with col_g1:
            st.subheader("Visualización del Caso I (Flechas Verticales)")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=x_izq_canvas, y=y_canvas, mode='lines', line=dict(color='#2c3e50', width=2)))
            fig1.add_trace(go.Scatter(x=x_der_canvas, y=y_canvas, mode='lines', line=dict(color='#2c3e50', width=2)))
            fig1.add_trace(go.Scatter(x=coordenadas_x_region, y=coordenadas_y_region, fill='toself', fillcolor='rgba(52, 152, 219, 0.25)', line=dict(color='rgba(255,255,255,0)')))
            
            # Flechas verticales de barrido dy
            pasos_x = np.linspace(ax_min + (bx_max-ax_min)*0.15, ax_min + (bx_max-ax_min)*0.85, 5)
            for px in pasos_x:
                try:
                    y_inf_val = float(g1_inf.subs(x, px).evalf())
                    y_sup_val = float(g2_sup.subs(x, px).evalf())
                    fig1.add_trace(go.Scatter(x=[px, px], y=[y_inf_val, y_sup_val], mode='lines+markers',
                                              marker=dict(symbol=['circle', 'arrow-up'], size=[4, 10], color='#e74c3c'),
                                              line=dict(color='#e74c3c', width=2)))
                except: pass
                
            fig1.update_layout(xaxis_title="Eje X", yaxis_title="Eje Y", template="plotly_white", height=450, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

        # --- FIGURA 2: BARRIDO HORIZONTAL ---
        with col_g2:
            st.subheader("Visualización del Caso II (Flechas Horizontales)")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=x_izq_canvas, y=y_canvas, mode='lines', line=dict(color='#2c3e50', width=2)))
            fig2.add_trace(go.Scatter(x=x_der_canvas, y=y_canvas, mode='lines', line=dict(color='#2c3e50', width=2)))
            fig2.add_trace(go.Scatter(x=coordenadas_x_region, y=coordenadas_y_region, fill='toself', fillcolor='rgba(46, 204, 113, 0.25)', line=dict(color='rgba(255,255,255,0)')))
            
            # Flechas horizontales de barrido dx
            pasos_y = np.linspace(cy_min + (dy_max-cy_min)*0.15, cy_min + (dy_max-cy_min)*0.85, 5)
            for py in pasos_y:
                x_izq_val = float(h1_izq.subs(y, py).evalf())
                x_der_val = float(h2_der.subs(y, py).evalf())
                fig2.add_trace(go.Scatter(x=[x_izq_val, x_der_val], y=[py, py], mode='lines+markers',
                                          marker=dict(symbol=['circle', 'arrow-right'], size=[4, 10], color='#e67e22'),
                                          line=dict(color='#e67e22', width=2)))
                
            fig2.update_layout(xaxis_title="Eje X", yaxis_title="Eje Y", template="plotly_white", height=450, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Error de procesamiento matemático: {e}. Comprueba la consistencia de las curvas e intenta de nuevo.")
