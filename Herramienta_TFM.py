import streamlit as st
import pydeck as pdk
import pandas as pd


st.set_page_config(
    page_title="Redistribución Turística",
    page_icon="🌍",
    layout="wide"
)

# ---- CARGAR DATOS EXCEL ----
@st.cache_data
def cargar_datos():
    xls = pd.ExcelFile("./Data_Dataestur/DATA_TOTAL.xlsx")
    df_total = pd.read_excel(xls, sheet_name="Total")
    df_coords = pd.read_excel(xls, sheet_name="Coordenadas ZT")
    return df_total, df_coords

df_total, df_coords = cargar_datos()

# ---- NORMALIZAR CLAVES PARA EL JOIN ----
df_total["ZONA_TURISTICA"] = df_total["ZONA_TURISTICA"].str.strip()
df_coords["ZONA_TURISTICA"] = df_coords["ZONA_TURISTICA"].str.strip()

# ---- UNIR COORDENADAS A DATOS ----
df = pd.merge(df_total, df_coords[["ZONA_TURISTICA", "lat", "long"]], on="ZONA_TURISTICA", how="left")

# ---- Encabezado con logo a la izquierda y texto a la derecha ----
col_logo, col_text, col_logo2 = st.columns([1, 4, 1])

with col_logo:
    st.image("Redisstour.svg", width=300)  # Ajusta el tamaño del logo a tu gusto

with col_logo2:
    st.image("Logo_Deep5.svg", width=300)  # Ajusta el tamaño del logo a tu gusto

with col_text:
    st.markdown("""
        <style>
            .header-container {
                display: flex;
                flex-direction: column;
                justify-content: center;
                height: 100%;
                font-family: 'Segoe UI', sans-serif;
            }
            .header-container h1 {
                font-size: 2.4rem;
                font-weight: 700;
                color: #1c1c1c;
                margin: 0;
            }
            .header-container p {
                font-size: 1.1rem;
                color: #5a5a5a;
                margin: 0 0.3rem 0 0;
            }
        </style>

        <div class="header-container">
            <h1>Redistribución Inteligente del Flujo Turístico</h1>
            <p>Hacia un turismo más sostenible y equilibrado en España</p>
        </div>
    """, unsafe_allow_html=True)

    


st.markdown("<br>", unsafe_allow_html=True)

# ---- LISTA DE IMÁGENES ----
imagenes = [
    {
        "url": "https://images.unsplash.com/photo-1605654464243-3668a4c0de3d?q=80&w=1700&auto=format&fit=crop&ixlib=rb-4.1.0",
        "ciudad": "Alhambra, Granada"
    },
    {
        "url": "https://images.unsplash.com/photo-1655405927893-96a5b68490c1?q=80&w=1548&auto=format&fit=crop&ixlib=rb-4.1.0",
        "ciudad": "Benidorm, Alicante"
    },
    {
        "url": "https://images.unsplash.com/photo-1536075597888-91fe9f9cacd7?q=80&w=1740&auto=format&fit=crop&ixlib=rb-4.1.0",
        "ciudad": "Mirador del Cap de la Barra, Costa Brava"
    },
    {
        "url": "https://images.unsplash.com/photo-1677939217436-01d7c0b8738e?q=80&w=1740&auto=format&fit=crop&ixlib=rb-4.1.0",
        "ciudad": "Torre del Oro, Sevilla"
    }
]

# ---- INDEXADO EN SESSION STATE ----
if "imagen_idx" not in st.session_state:
    st.session_state.imagen_idx = 0

if "seccion" not in st.session_state:
    st.session_state.seccion = "Inicio"

# ---- FUNCIONES DE BOTONES ----
def avanzar():
    st.session_state.imagen_idx = (st.session_state.imagen_idx + 1) % len(imagenes)

def retroceder():
    st.session_state.imagen_idx = (st.session_state.imagen_idx - 1) % len(imagenes)


# ---- MENÚ DE FUNCIONALIDADES (barra centrada y estirada) ----
col_left, col0, col1, col2, col3, col4, col_right = st.columns([1, 2, 2, 2, 2, 2, 1])

with col0:
    if st.button("Inicio", use_container_width=True):
        st.session_state.seccion = "Inicio"
with col1:
    if st.button("Destino alternativo", use_container_width=True):
        st.session_state.seccion = "Seleccionar destino alternativo"
with col2:
    if st.button("Mapa saturación", use_container_width=True):
        st.session_state.seccion = "Ver mapas de saturación"
with col3:
    if st.button("Datos históricos", use_container_width=True):
        st.session_state.seccion = "Consultar datos históricos"
with col4:
    if st.button("Acerca del proyecto", use_container_width=True):
        st.session_state.seccion = "Acerca del proyecto"



# Mostrar contenido según lo seleccionado
opcion = st.session_state.seccion

# ---- SECCIONES SEGÚN OPCIÓN ----
if opcion == "Inicio":
        
        # ---- CSS personalizado ----
        st.markdown("""
            <style>
                .img-wrapper {
                    position: relative;
                    width: 100%;
                    height: 450px;
                }
                .img-wrapper img {
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                    border-radius: 5px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                }
                .city-label {
                    position: absolute;
                    bottom: 20px;
                    right: 30px;
                    background: rgba(0, 0, 0, 0.5);
                    color: white;
                    padding: 0.5rem 1rem;
                    border-radius: 6px;
                    font-size: 1rem;
                    font-weight: 500;
                }
            </style>
        """, unsafe_allow_html=True)


        # Mostrar la imagen
        imagen_actual = imagenes[st.session_state.imagen_idx]
        st.markdown(f"""
            <div class="img-wrapper">
                <img src="{imagen_actual['url']}" alt="{imagen_actual['ciudad']}">
                <div class="city-label">{imagen_actual['ciudad']}</div>
            </div>
        """, unsafe_allow_html=True)


        st.markdown("<br>", unsafe_allow_html=True)

        # Crear columnas para los botones y un espacio central
        col1, col2, col3 = st.columns([1, 30, 1])

        # Botón de retroceder en la primera columna
        if col1.button('◀'):
            retroceder()

        # Botón de avanzar en la segunda columna
        if col3.button('▶'):
            avanzar()

        
        # ---- MOTIVACIÓN DEL PROYECTO ----
        st.subheader("Motivación del proyecto")
        st.info("España se enfrenta al reto de gestionar el crecimiento turístico sin comprometer la sostenibilidad ni la calidad de vida de sus habitantes.")

        st.write("""
        En las últimas décadas, España se ha consolidado como uno de los destinos turísticos más importantes del mundo. 
        Sin embargo, este crecimiento ha generado desafíos significativos:

        - **Saturación de destinos populares** como Barcelona, Sevilla o Ibiza, especialmente en temporada alta.
        - **Presión sobre las infraestructuras locales**, recursos y servicios.
        - **Impacto ambiental y pérdida de calidad en la experiencia turística**.

        A pesar de la concentración turística en zonas muy concretas, existen numerosos destinos con alto potencial que permanecen infrautilizados.  
        Esto evidencia la necesidad de redistribuir de forma más equitativa el flujo turístico en el territorio nacional.
        """)

        # ---- OBJETIVO DEL SISTEMA ----
        st.subheader("Objetivo del sistema")
        st.info("Desarrollar una herramienta inteligente para analizar, visualizar y redistribuir el turismo de forma sostenible en España.")

        st.write("""
        Este proyecto tiene como propósito el diseño de una plataforma interactiva basada en datos, con los siguientes objetivos clave:

        - Analizar datos históricos para detectar **zonas turísticas saturadas**.
        - Sugerir **destinos alternativos** menos masificados con características similares.
        - Proporcionar **visualización dinámica** de la evolución turística mediante mapas y gráficos.
        - Facilitar la toma de decisiones a gestores públicos y privados, con una visión centrada en la **sostenibilidad y la equidad territorial**.

        Este sistema se alinea con los principios de los **Objetivos de Desarrollo Sostenible** y las estrategias de **turismo inteligente en Europa**.
        """)


if opcion == "Seleccionar destino alternativo":
    st.subheader("🔍 Recomendador de destinos turísticos alternativos")
    st.info("Aquí podrás introducir tu destino actual y ver sugerencias menos saturadas con características similares.")
    # Aquí irá la lógica de recomendación futur


elif opcion == "Ver mapas de saturación":
    st.subheader("Mapa de saturación turística por zona")
    st.info("Visualiza la concentración de turistas en cada zona mediante columnas proporcionales al número de viajeros.")

    # ---- FILTROS INTERACTIVOS ----
    st.markdown("### 🎚️ Filtros temporales y tipo de turismo")

    col_f1, col_f2, col_f3 = st.columns([1.2, 1.2, 2])

    # Año
    with col_f1:
        años_disponibles = sorted(df["AÑO"].dropna().unique())
        opciones_año = ["Todos los años"] + [str(a) for a in años_disponibles]
        año_seleccionado = st.selectbox("📅 Año", opciones_año)

    # Mes
    with col_f2:
        meses_nombres = {
            1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
            7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
        }
        meses_disponibles = sorted(df["MES"].dropna().unique())
        opciones_mes = ["Todos los meses"] + [meses_nombres[m] for m in meses_disponibles]
        mes_seleccionado = st.selectbox("🗓️ Mes", opciones_mes)

    # Tipo de turismo
    with col_f3:
        tipo = st.selectbox("🏨 Tipo de turismo", [
            "Turismo Hotelero", "Turismo Rural", "Apartamentos", "Campings"
        ])

    # ---- ASIGNAR COLUMNA CORRESPONDIENTE ----
    columnas_tipo = {
        "Turismo Hotelero": "VIAJEROS_EOH",
        "Turismo Rural": "VIAJEROS_EOTR",
        "Apartamentos": "VIAJEROS_EOAP",
        "Campings": "VIAJEROS_EOAC"
    }
    columna_viajeros = columnas_tipo[tipo]

    # ---- APLICAR FILTROS SEGÚN SELECCIONES ----
    df_filtrado = df.copy()
    if año_seleccionado != "Todos los años":
        df_filtrado = df_filtrado[df_filtrado["AÑO"] == int(año_seleccionado)]
    if mes_seleccionado != "Todos los meses":
        mes_num = [k for k, v in meses_nombres.items() if v == mes_seleccionado][0]
        df_filtrado = df_filtrado[df_filtrado["MES"] == mes_num]

    # Filtrar valores no nulos y mayores que 0
    df_filtrado = df_filtrado[df_filtrado[columna_viajeros] > 0]

    # ---- AGRUPAR POR ZONA Y COORDENADAS ----
    df_grouped = df_filtrado.groupby(["ZONA_TURISTICA", "lat", "long"], as_index=False)[columna_viajeros].sum()
    df_grouped = df_grouped.rename(columns={columna_viajeros: "viajeros"})

    # ---- MOSTRAR MAPA CON COLUMNAS ----
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=39,
            longitude=-3.5,
            zoom=3.5,
            pitch=30
        ),
        layers=[
            pdk.Layer(
                "ColumnLayer",
                data=df_grouped,
                get_position='[long, lat]',
                get_elevation='viajeros',
                elevation_scale=0.1,
                radius=10000,
                get_fill_color='[200, 30, 0, 160]',
                pickable=True,
                auto_highlight=True,
            )
        ],
        tooltip={
            "html": """
                <div style='font-family: Segoe UI; font-size: 13px;'>
                    <strong>{ZONA_TURISTICA}</strong><br>
                    <span style='color: black;'>Viajeros: {viajeros}</span>
                </div>
            """,
            "style": {
                "backgroundColor": "#f9f9f9",
                "color": "#000",
                "border": "1px solid #ccc",
                "borderRadius": "6px",
                "padding": "10px"
            }
        }

    ))




    
elif opcion == "Consultar datos históricos":
    st.subheader("📈 Evolución de datos turísticos")
    st.info("Consulta viajeros, pernoctaciones y ocupación a lo largo del tiempo.")
    # Aquí irán gráficos y tablas

elif opcion == "Acerca del proyecto":
    st.subheader("ℹ️ Acerca de este proyecto")
    st.write("""
Este proyecto forma parte del Trabajo de Fin de Máster (TFM) en el que se desarrolla un sistema
de recomendación y análisis para mejorar la distribución del turismo en España, utilizando
herramientas de ciencia de datos, visualización interactiva y modelado predictivo.
    """)

# ---- PIE DE PÁGINA ----
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2025 Proyecto TFM - Redistribución Turística | Deep5</p>", unsafe_allow_html=True)
