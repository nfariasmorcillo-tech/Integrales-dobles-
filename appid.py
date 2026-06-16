import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go

# Configuración de la página web
st.set_page_config(page_title="Módulo de Integrales Dobles", page_icon="∬", layout="wide")

st.title("∬ Laboratorio de Integrales Dobles: Regiones Generales")
st.markdown("Analiza regiones de integración $R$, determina límites automáticamente y compara el **Caso I (Barrido Vertical)** con el **Caso II (Barrido Horizontal)**.")

# =========================================================================
# BARRA LATERAL: INGRESO DE DATOS
# =========================================================================
st.sidebar.header("📥 Parámetros de la Integral")

func_str = st.sidebar.text_input("Función f(x, y):", value="x * y")
eqA_str = st.sidebar.text_input("Curva Frontera A (y=... o x=...):", value="y + 1")
eqB_str = st.sidebar.text_input("Curva Frontera B (y=... o x=...):", value="(y**2 - 6) / 2")

st.sidebar.divider()
st.sidebar.markdown(
    "**Sugerencia Didáctica:**\n"
    "El sistema autodetecta las variables, valida algebraicamente las ramas de raíces cuadradas "
    "mediante comprobación cruzada en los puntos medios y filtra discontinuidades asintóticas."
)

calcular = st.sidebar.button("Calcular Región e Integrales", type="primary")

# =========================================================================
# MOTOR ALGEBRAICO ADAPTATIVO CON VALIDACIÓN CRUZADA
# =========================================================================
if calcular:
    try:
        x, y = sp.symbols('x y')
        f_xy = sp.sympify(func_str)
        
        rawA = sp.sympify(eqA_str)
        rawB = sp.sympify(eqB_str)

        sols_x = []
        try: sols_x = [float(pt.evalf()) for pt in sp.solve(sp.Eq(rawA, rawB), x) if pt.is_real]
        except: pass

        sols_y = []
        try: sols_y = [float(pt.evalf()) for pt in sp.solve(sp.Eq(rawA, rawB), y) if pt.is_real]
        except: pass

        # Decisión del eje base de anclaje de la región acotada
        if len(sols_y) >= 2:
            cy_min, dy_max = min(sols_y), max(sols_y)
            y_medio = (cy_min + dy_max) / 2.0
            
            valA = float(rawA.subs(y, y_medio).evalf())
            valB = float(rawB.subs(y, y_medio).evalf())
            
            h2_der = rawA if valA >= valB else rawB
            h1_izq = rawB if valA >= valB else rawA
            
            ax_min = min(float(h1_izq.subs(y, cy_min).evalf()), float(h1_izq.subs(y, dy_max).evalf()))
            bx_max = max(float(h2_der.subs(y, cy_min).evalf()), float(h2_der.subs(y, dy_max).evalf()))
            
            x_medio = (ax_min + bx_max) / 2.0
            ramas_izq = sp.solve(sp.Eq(x, h1_izq), y)
            ramas_der = sp.solve(sp.Eq(x, h2_der), y)
            
            todas_las_ramas = ramas_izq + ramas_der
            ramas_validas = []
            for r in todas_las_ramas:
                try:
                    val_r = r.subs(x, x_medio)
                    if val_r.is_real and (cy_min <= float(val_r.evalf()) <= dy_max):
                        ramas_validas.append((float(val_r.evalf()), r))
                except: pass
            
            ramas_validas = sorted(list(set(ramas_validas)), key=lambda t: t[0])
            
            if len(ramas_validas) >= 2:
                g1_inf = ramas_validas[0][1]
                g2_sup = ramas_validas[-1][1]
            else:
                g1_inf = sp.solve(sp.Eq(x, h2_der), y)[0]
                g2_sup = sp.solve(sp.Eq(x, h1_izq), y)[0]
                if g1_inf.subs(x, x_medio).evalf() > g2_sup.subs(x, x_medio).evalf():
                    g1_inf, g2_sup = g2_sup, g1_inf

        else:
            ax_min, bx_max = min(sols_x), max(sols_x)
            x_medio = (ax_min + bx_max) / 2.0
            
            valA = float(rawA.subs(x, x_medio).evalf())
            valB = float(rawB.subs(x, x_medio).evalf())
            
            g2_sup = rawA if valA >= valB else rawB
            g1_inf = rawB if valA >= valB else rawA
            
            cy_min = min(float(g1_inf.subs(x, ax_min).evalf()), float(g1_inf.subs(x, bx_max).evalf()))
            dy_max = max(float(g2_sup.subs(x, ax_min).evalf()), float(g2_sup.subs(x, bx_max).evalf()))
            
            y_medio = (cy_min + dy_max) / 2.0
            ramas_inf = sp.solve(sp.Eq(y, g1_inf), x)
            ramas_sup = sp.solve(sp.Eq(y, g2_sup), x)
            
            todas_las_ramas = ramas_inf + ramas_sup
            ramas_validas = []
            for r in todas_las_ramas:
                try:
                    val_r = r.subs(y, y_medio)
                    if val_r.is_real and (ax_min <= float(val_r.evalf()) <= bx_max):
                        ramas_validas.append((float(val_r.evalf()), r))
                except: pass
            
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
        # RENDERIZADO MATEMÁTICO DESGLOSADO
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
        # GRÁFICOS INTERACTIVOS CON FILTRADO COMPLEJO Y ASÍNTOTAS
        # =========================================================================
        st.header("2. Visualización Pedagógica del Sentido de Barrido")
        col_g1, col_g2 = st.columns(2)

        # Función de apoyo para limpieza numérica y prevención de saltos por asíntotas complejas
        def evaluar_limpio(expr, var, val):
            try:
                res = expr.subs(var, val).evalf()
                res_f = float(res)
                if np.isnan(res_f) or np.isinf(res_f) or abs(res_f) > 1e4:
                    return None
                return res_f
            except:
                return None

        y_mesh = np.linspace(cy_min, dy_max, 200)
        x_izq_vals, x_der_vals, y_mesh_clean = [], [], []

        for v in y_mesh:
            xi = evaluar_limpio(h1_izq, y, v)
            xd = evaluar_limpio(h2_der, y, v)
            if xi is not None and xd is not None:
                x_izq_vals.append(xi)
                x_der_vals.append(xd)
                y_mesh_clean.append(v)

        coordenadas_x_region = np.concatenate([x_izq_vals, x_der_vals[::-1]])
        coordenadas_y_region = np.concatenate([y_mesh_clean, y_mesh_clean[::-1]])

        y_canvas = np.linspace(cy_min - 0.5, dy_max + 0.5, 250)
        x_izq_canvas, x_der_canvas, y_canvas_clean = [], [], []
        for v in y_canvas:
            xi = evaluar_limpio(h1_izq, y, v)
            xd = evaluar_limpio(h2_der, y, v)
            if xi is not None and xd is not None:
                x_izq_canvas.append(xi)
                x_der_canvas.append(xd)
                y_canvas_clean.append(v)

        # --- FIGURA 1: BARRIDO VERTICAL ---
        with col_g1:
            st.subheader("Visualización del Caso I (Flechas Verticales)")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=x_izq_canvas, y=y_canvas_clean, mode='lines', line=dict(color='#2c3e50', width=2)))
            fig1.add_trace(go.Scatter(x=x_der_canvas, y=y_canvas_clean, mode='lines', line=dict(color='#2c3e50', width=2)))
            fig1.add_trace(go.Scatter(x=coordenadas_x_region, y=coordenadas_y_region, fill='toself', fillcolor='rgba(52, 152, 219, 0.25)', line=dict(color='rgba(255,255,255,0)')))
            
            pasos_x = np.linspace(ax_min + (bx_max-ax_min)*0.15, ax_min + (bx_max-ax_min)*0.85, 5)
            for px in pasos_x:
                yi = evaluar_limpio(g1_inf, x, px)
                ys = evaluar_limpio(g2_sup, x, px)
                if yi is not None and ys is not None:
                    fig1.add_trace(go.Scatter(x=[px, px], y=[yi, ys], mode='lines+markers',
                                              marker=dict(symbol=['circle', 'arrow-up'], size=[4, 10], color='#e74c3c'),
                                              line=dict(color='#e74c3c', width=2)))
                
            fig1.update_layout(xaxis_title="Eje X", yaxis_title="Eje Y", template="plotly_white", height=450, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

        # --- FIGURA 2: BARRIDO HORIZONTAL ---
        with col_g2:
            st.subheader("Visualización del Caso II (Flechas Horizontales)")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=x_izq_canvas, y=y_canvas_clean, mode='lines', line=dict(color='#2c3e50', width=2)))
            fig2.add_trace(go.Scatter(x=x_der_canvas, y=y_canvas_clean, mode='lines', line=dict(color='#2c3e50', width=2)))
            fig2.add_trace(go.Scatter(x=coordenadas_x_region, y=coordenadas_y_region, fill='toself', fillcolor='rgba(46, 204, 113, 0.25)', line=dict(color='rgba(255,255,255,0)')))
            
            pasos_y = np.linspace(cy_min + (dy_max-cy_min)*0.15, cy_min + (dy_max-cy_min)*0.85, 5)
            for py in pasos_y:
                x_izq_val = evaluar_limpio(h1_izq, y, py)
                x_der_val = evaluar_limpio(h2_der, y, py)
                if x_izq_val is not None and x_der_val is not None:
                    fig2.add_trace(go.Scatter(x=[x_izq_val, x_der_val], y=[py, py], mode='lines+markers',
                                              marker=dict(symbol=['circle', 'arrow-right'], size=[4, 10], color='#e67e22'),
                                              line=dict(color='#e67e22', width=2)))
                
            fig2.update_layout(xaxis_title="Eje X", yaxis_title="Eje Y", template="plotly_white", height=450, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Error de procesamiento matemático: {e}. Comprueba la consistencia de las curvas.")
