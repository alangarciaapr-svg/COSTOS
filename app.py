import streamlit as st
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Calculadora Costos Maquinaria", layout="wide")

st.title("🚜 Calculadora de Costos Operacionales y Mantención")
st.markdown("Comparativa de costos mensuales basada en horas de trabajo y prorrateo de mantenciones.")

# --- BARRA LATERAL (CONFIGURACIÓN GLOBAL) ---
with st.sidebar:
    st.header("1. Parámetros Globales")
    valor_uf = st.number_input("Valor UF Hoy (CLP)", value=37000, step=100)
    precio_petroleo = st.number_input("Precio Litro Petróleo (CLP)", value=1000, step=10)
    
    st.markdown("---")
    st.header("2. Configuración Máquinas")
    
    # Máquina 1
    st.subheader("Máquina 1")
    nombre_m1 = st.text_input("Nombre M1", value="Máquina A")
    horas_m1 = st.number_input(f"Horas Mensuales {nombre_m1}", value=400, min_value=1)
    
    # Máquina 2
    st.subheader("Máquina 2")
    nombre_m2 = st.text_input("Nombre M2", value="Máquina B")
    horas_m2 = st.number_input(f"Horas Mensuales {nombre_m2}", value=400, min_value=1)

# --- FUNCIÓN DE CÁLCULO ---
def calcular_item(tipo, input_m1, input_m2, factor=1, intervalo=None):
    """
    tipo: 'fijo', 'uf', 'petroleo', 'insumo', 'semanal', 'mantencion'
    input_m1/m2: Valores ingresados por usuario
    factor: Multiplicador (ej. 5 cadenas, 10 tubos)
    intervalo: Horas de mantención (ej. 600, 10000)
    """
    costo_m1 = 0
    costo_m2 = 0
    
    # Lógica de cálculo según el tipo de ítem
    if tipo == 'fijo': # Valor mensual directo
        costo_m1 = input_m1
        costo_m2 = input_m2
        
    elif tipo == 'uf': # Valor en UF a convertir
        costo_m1 = input_m1 * valor_uf
        costo_m2 = input_m2 * valor_uf
        
    elif tipo == 'petroleo': # Litros diarios a costo mensual (30 días)
        costo_m1 = input_m1 * 30 * precio_petroleo
        costo_m2 = input_m2 * 30 * precio_petroleo
        
    elif tipo == 'insumo': # Cantidad fija mensual (ej: 5 cadenas * precio unitario)
        costo_m1 = input_m1 * factor
        costo_m2 = input_m2 * factor
        
    elif tipo == 'semanal': # Litros semanales a mensual (x4.3 semanas)
        # Input es el PRECIO DEL LITRO en este caso, factor es la cantidad
        costo_m1 = (factor * 4.3) * input_m1
        costo_m2 = (factor * 4.3) * input_m2
        
    elif tipo == 'mantencion': # Prorrateo: (Costo Evento / Intervalo) * Horas Mes
        if intervalo:
            costo_m1 = (input_m1 / intervalo) * horas_m1
            costo_m2 = (input_m2 / intervalo) * horas_m2

    return int(costo_m1), int(costo_m2)

# --- INTERFAZ DE INGRESO DE DATOS ---
st.header("3. Ingreso de Costos")

# Listas para guardar los resultados
datos_tabla = []

# Agrupador 1: Arriendo y Operación
with st.expander("A. Arriendo, Personal y Combustible", expanded=True):
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1: st.markdown("**Ítem**")
    with col2: st.markdown(f"**{nombre_m1}**")
    with col3: st.markdown(f"**{nombre_m2}**")

    # Arriendo
    val_arr_1 = col2.number_input("Arriendo - Valor Mensual M1", min_value=0, key="arr1")
    val_arr_2 = col3.number_input("Arriendo - Valor Mensual M2", min_value=0, key="arr2")
    c1, c2 = calcular_item('fijo', val_arr_1, val_arr_2)
    datos_tabla.append(["Arriendo", c1, c2])

    # Operadores
    val_op1_1 = col2.number_input("Operador Turno 1 - Sueldo M1", min_value=0, key="op1_1")
    val_op1_2 = col3.number_input("Operador Turno 1 - Sueldo M2", min_value=0, key="op1_2")
    c1, c2 = calcular_item('fijo', val_op1_1, val_op1_2)
    datos_tabla.append(["Operador Turno 1", c1, c2])

    val_op2_1 = col2.number_input("Operador Turno 2 - Sueldo M1", min_value=0, key="op2_1")
    val_op2_2 = col3.number_input("Operador Turno 2 - Sueldo M2", min_value=0, key="op2_2")
    c1, c2 = calcular_item('fijo', val_op2_1, val_op2_2)
    datos_tabla.append(["Operador Turno 2", c1, c2])

    # Petróleo (Litros)
    litros_t1_1 = col2.number_input("Petróleo T1 - Litros Diarios M1", min_value=0.0, key="pet1_1")
    litros_t1_2 = col3.number_input("Petróleo T1 - Litros Diarios M2", min_value=0.0, key="pet1_2")
    c1, c2 = calcular_item('petroleo', litros_t1_1, litros_t1_2)
    datos_tabla.append(["Petróleo Turno 1", c1, c2])

    litros_t2_1 = col2.number_input("Petróleo T2 - Litros Diarios M1", min_value=0.0, key="pet2_1")
    litros_t2_2 = col3.number_input("Petróleo T2 - Litros Diarios M2", min_value=0.0, key="pet2_2")
    c1, c2 = calcular_item('petroleo', litros_t2_1, litros_t2_2)
    datos_tabla.append(["Petróleo Turno 2", c1, c2])
    
    # Seguro UF
    seg_1 = col2.number_input("Seguro RENTA (UF) M1", min_value=0.0, key="seg1")
    seg_2 = col3.number_input("Seguro RENTA (UF) M2", min_value=0.0, key="seg2")
    c1, c2 = calcular_item('uf', seg_1, seg_2)
    datos_tabla.append(["Seguro RENTA", c1, c2])

# Agrupador 2: Insumos
with st.expander("B. Insumos (Cadenas, Espadas, Aceites)"):
    col1, col2, col3 = st.columns([2, 1, 1])
    
    # Cadenas (x5)
    cad1_1 = col2.number_input("Costo Unitario Cadena M1 (se calc. x5)", min_value=0, key="cad1")
    cad1_2 = col3.number_input("Costo Unitario Cadena M2 (se calc. x5)", min_value=0, key="cad2")
    c1, c2 = calcular_item('insumo', cad1_1, cad1_2, factor=5)
    datos_tabla.append(["Cadenas Turno 1 (5 un)", c1, c2])
    
    cad2_1 = col2.number_input("Costo Unitario Cadena T2 M1 (se calc. x5)", min_value=0, key="cad2_1")
    cad2_2 = col3.number_input("Costo Unitario Cadena T2 M2 (se calc. x5)", min_value=0, key="cad2_2")
    c1, c2 = calcular_item('insumo', cad2_1, cad2_2, factor=5)
    datos_tabla.append(["Cadenas Turno 2 (5 un)", c1, c2])

    # Espadas (x1.5)
    esp1_1 = col2.number_input("Costo Unitario Espada T1 M1 (se calc. x1.5)", min_value=0, key="esp1")
    esp1_2 = col3.number_input("Costo Unitario Espada T1 M2 (se calc. x1.5)", min_value=0, key="esp2")
    c1, c2 = calcular_item('insumo', esp1_1, esp1_2, factor=1.5)
    datos_tabla.append(["Espadas Turno 1 (1.5 un)", c1, c2])
    
    esp2_1 = col2.number_input("Costo Unitario Espada T2 M1 (se calc. x1.5)", min_value=0, key="esp2_1")
    esp2_2 = col3.number_input("Costo Unitario Espada T2 M2 (se calc. x1.5)", min_value=0, key="esp2_2")
    c1, c2 = calcular_item('insumo', esp2_1, esp2_2, factor=1.5)
    datos_tabla.append(["Espadas Turno 2 (1.5 un)", c1, c2])

    # Grasa (x10)
    grasa_1 = col2.number_input("Costo Unitario Tubo Grasa M1 (se calc. x10)", min_value=0, key="gras1")
    grasa_2 = col3.number_input("Costo Unitario Tubo Grasa M2 (se calc. x10)", min_value=0, key="gras2")
    c1, c2 = calcular_item('insumo', grasa_1, grasa_2, factor=10)
    datos_tabla.append(["Grasa (10 tubos)", c1, c2])
    
    # Aceite Hidraulico (19 lt semanal)
    # Aquí pedimos precio por litro
    aceite_1 = col2.number_input("Precio Litro Aceite Hidr. M1 (Consumo 19L sem)", min_value=0, key="oil1")
    aceite_2 = col3.number_input("Precio Litro Aceite Hidr. M2 (Consumo 19L sem)", min_value=0, key="oil2")
    c1, c2 = calcular_item('semanal', aceite_1, aceite_2, factor=19)
    datos_tabla.append(["Aceite Hidráulico (19L/sem)", c1, c2])

# Agrupador 3: Mantenciones Periódicas
with st.expander("C. Mantenciones (Prorrateo Mensual según Horas)", expanded=True):
    st.info("Ingrese el VALOR TOTAL del evento de mantención. El sistema calculará el costo mensual proporcional.")
    col1, col2, col3 = st.columns([2, 1, 1])

    # Lista de mantenciones con sus intervalos
    mantenciones = [
        ("Mantención 600 horas", 600),
        ("Mantención 1200 horas", 1200),
        ("Mantención 1800 horas", 1800),
        ("Mantención Hidráulica 6000 hrs", 6000),
        ("Mantención Grúa 10000 hrs", 10000),
        ("Sist. Electrónico Cabezal 10000 hrs", 10000),
        ("Sist. Electrónico Base 10000 hrs", 10000),
        ("Mantención Mecánica 10000 hrs", 10000),
    ]

    for nombre, intervalo in mantenciones:
        m1_val = col2.number_input(f"Valor Total {nombre} M1", min_value=0, key=f"m_{intervalo}_{nombre}_1")
        m2_val = col3.number_input(f"Valor Total {nombre} M2", min_value=0, key=f"m_{intervalo}_{nombre}_2")
        c1, c2 = calcular_item('mantencion', m1_val, m2_val, intervalo=intervalo)
        datos_tabla.append([nombre, c1, c2])

# Agrupador 4: Largo Plazo / Overhaul (20.000 hrs)
with st.expander("D. Overhaul y Componentes Mayores (20.000 horas)", expanded=True):
    col1, col2, col3 = st.columns([2, 1, 1])
    
    overhauls = [
        ("Neumáticos 20000 hrs", 20000),
        ("Valtras 20000 hrs", 20000),
        ("Overhaul Tren Motriz (5 años)", 20000),
        ("Overhaul Cambio Motor (5 años)", 20000),
    ]
    
    for nombre, intervalo in overhauls:
        m1_val = col2.number_input(f"Valor Total {nombre} M1", min_value=0, key=f"ov_{nombre}_1")
        m2_val = col3.number_input(f"Valor Total {nombre} M2", min_value=0, key=f"ov_{nombre}_2")
        c1, c2 = calcular_item('mantencion', m1_val, m2_val, intervalo=intervalo)
        datos_tabla.append([nombre, c1, c2])

# --- RESULTADOS ---
st.markdown("---")
st.header("4. Resumen de Costos Mensuales")

# Crear DataFrame
df = pd.DataFrame(datos_tabla, columns=["Ítem", f"Costo Mes {nombre_m1}", f"Costo Mes {nombre_m2}"])

# Calcular Totales
total_m1 = df[f"Costo Mes {nombre_m1}"].sum()
total_m2 = df[f"Costo Mes {nombre_m2}"].sum()

# Formato moneda para mostrar en tabla
df_mostrar = df.copy()
df_mostrar[f"Costo Mes {nombre_m1}"] = df_mostrar[f"Costo Mes {nombre_m1}"].apply(lambda x: f"${x:,.0f}")
df_mostrar[f"Costo Mes {nombre_m2}"] = df_mostrar[f"Costo Mes {nombre_m2}"].apply(lambda x: f"${x:,.0f}")

st.dataframe(df_mostrar, use_container_width=True)

# Métricas Finales
c1, c2 = st.columns(2)
with c1:
    st.metric(label=f"Costo Total Mensual {nombre_m1}", value=f"${total_m1:,.0f}")
    if horas_m1 > 0:
        costo_hora_1 = total_m1 / horas_m1
        st.caption(f"Costo por Hora: ${costo_hora_1:,.0f}")

with c2:
    st.metric(label=f"Costo Total Mensual {nombre_m2}", value=f"${total_m2:,.0f}")
    if horas_m2 > 0:
        costo_hora_2 = total_m2 / horas_m2
        st.caption(f"Costo por Hora: ${costo_hora_2:,.0f}")
