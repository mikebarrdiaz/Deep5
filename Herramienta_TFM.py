import streamlit as st
import pydeck as pdk
import pandas as pd

# =========================
# CONFIG BASICA DE LA APP
# =========================
st.set_page_config(
    page_title="Redistribución Turística",
    page_icon="🌍",
    layout="wide"
)

# =========================
# PALETA + UTILIDADES
# =========================
COLORS = {
    "indigo_dye": "#224762",
    "paynes_gray": "#385971",
    "anti_flash_white": "#e6eaed",
    "slate_gray": "#688194",
    "lapis_lazuli": "#306388",
}

def hex_to_rgba(hex_str, alpha=1.0):
    hex_str = hex_str.strip("#")
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    a = int(max(0, min(1, alpha)) * 255)
    return [r, g, b, a]


# =========================
# CSS GLOBAL (fondo blanco + hero intacto + selects en paleta)
# =========================
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
:root {{
  --indigo_dye: {COLORS["indigo_dye"]};
  --paynes_gray: {COLORS["paynes_gray"]};
  --slate_gray: {COLORS["slate_gray"]};
  --lapis_lazuli: {COLORS["lapis_lazuli"]};
}}

html, body, .stApp {{
  background: #ffffff !important;
  color: var(--indigo_dye) !important;
  font-family: 'Inter', sans-serif !important;
}}

h1, h2, h3, h4, h5 {{
  color: var(--indigo_dye) !important;
}}

/* ---------------- Botones ---------------- */
.stButton > button {{
  background: var(--lapis_lazuli) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  padding: .5rem 1rem !important;
  box-shadow: 0 2px 8px rgba(0,0,0,.06);
  transition: background .15s ease, transform .05s ease;
}}
.stButton > button:hover {{ background: var(--paynes_gray) !important; }}
.stButton > button:active {{ transform: translateY(1px); }}

/* --------------- Inputs / Selects / Multiselects --------------- */
[data-baseweb="select"] > div,
.stSelectbox div[role="combobox"],
.stMultiSelect div[role="combobox"],
.stTextInput input {{
  border-radius: 6px !important;
  border: 1px solid var(--lapis_lazuli) !important;
  background-color: #fff !important;
  color: var(--indigo_dye) !important;
  box-shadow: none !important;
}}
/* Focus visible en la paleta */
[data-baseweb="select"] > div:focus-within,
.stSelectbox div[role="combobox"]:focus-within,
.stMultiSelect div[role="combobox"]:focus-within,
.stTextInput input:focus {{
  outline: 2px solid color-mix(in srgb, var(--lapis_lazuli) 60%, white) !important;
  border-color: var(--paynes_gray) !important;
}}

/* Etiquetas (chips) seleccionadas en MultiSelect */
.stMultiSelect [data-baseweb="tag"] {{
  background-color: var(--lapis_lazuli) !important;
  color: #fff !important;
  border-radius: 6px !important;
  padding: 0.15rem 0.45rem !important;
  border: none !important;
}}
.stMultiSelect [data-baseweb="tag"] * {{
  color: #fff !important;
}}
/* Dropdown del select (lista de opciones) */
div[role="listbox"] {{
  border: 1px solid var(--lapis_lazuli) !important;
  box-shadow: 0 6px 18px rgba(0,0,0,.08) !important;
}}
div[role="option"] {{
  color: var(--indigo_dye) !important;
}}
div[role="option"][aria-selected="true"],
div[role="option"]:hover {{
  background: color-mix(in srgb, var(--lapis_lazuli) 10%, white) !important;
}}

/* Etiquetas de los widgets */
.stSelectbox label, .stMultiSelect label, .stSlider label, .stTextInput label {{
  color: var(--slate_gray) !important;
}}

/* ---------------- Alertas ---------------- */
div.stAlert > div {{
  background: #f6f7f8;
  color: var(--indigo_dye);
  border-radius: 6px;
  border: 1px solid color-mix(in srgb, var(--slate_gray) 25%, white);
}}

/* ---------------- Footer ---------------- */
.footer-note {{
  text-align: center;
  color: var(--slate_gray);
  font-size: 0.9rem;
}}

/* ---------------- HERO (intacto) ---------------- */
.img-wrapper {{
  position: relative;
  width: 100%;
  height: 450px;
}}
.img-wrapper img {{
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 10px !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}}
.city-label {{
  position: absolute;
  bottom: 20px;
  right: 30px;
  background: rgba(56, 89, 113, 0.75); /* paynes gray */
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 6px;
  font-size: 1rem;
  font-weight: 500;
  backdrop-filter: blur(3px);
}}

/* ---------------- Header ---------------- */
.header-container {{
  display: flex;
  flex-direction: column;
  justify-content: center;
  height: 100%;
  font-family: 'Segoe UI', sans-serif;
}}
.header-container h1 {{
  font-size: 2.4rem;
  font-weight: 700;
  color: var(--indigo_dye);
  margin: 0;
}}
.header-container p {{
  font-size: 1.1rem;
  color: var(--slate_gray);
  margin: 0 0.3rem 0 0;
}}
</style>
""", unsafe_allow_html=True)



# =========================
# CARGA DE DATOS
# =========================
@st.cache_data
def cargar_datos():
    xls = pd.ExcelFile("./Data_Dataestur/DATA_TOTAL.xlsx")
    df_total = pd.read_excel(xls, sheet_name="Total")
    df_coords = pd.read_excel(xls, sheet_name="Coordenadas ZT")
    return df_total, df_coords

df_total, df_coords = cargar_datos()

# Normalizar claves
df_total["ZONA_TURISTICA"] = df_total["ZONA_TURISTICA"].astype(str).str.strip()
df_coords["ZONA_TURISTICA"] = df_coords["ZONA_TURISTICA"].astype(str).str.strip()

# Unir coordenadas
df = pd.merge(df_total, df_coords[["ZONA_TURISTICA", "lat", "long"]], on="ZONA_TURISTICA", how="left")

# =========================
# ENCABEZADO (logos + texto)
# =========================
col_logo, col_text, col_logo2 = st.columns([1, 4, 1])

with col_logo:
    st.image("./Logos/Redisstour.svg", width=300)

with col_logo2:
    st.image("./Logos/Logo_Deep5.svg", width=300)

with col_text:
    st.markdown("""
        <div class="header-container">
            <h1>Redistribución Inteligente del Flujo Turístico</h1>
            <p>Hacia un turismo más sostenible y equilibrado en España</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# IMAGENES HERO
# =========================
imagenes = [
    {"url":"https://images.unsplash.com/photo-1605654464243-3668a4c0de3d?q=80&w=1700&auto=format&fit=crop&ixlib=rb-4.1.0","ciudad":"Alhambra, Granada"},
    {"url":"https://images.unsplash.com/photo-1655405927893-96a5b68490c1?q=80&w=1548&auto=format&fit=crop&ixlib=rb-4.1.0","ciudad":"Benidorm, Alicante"},
    {"url":"https://images.unsplash.com/photo-1536075597888-91fe9f9cacd7?q=80&w=1740&auto=format&fit=crop&ixlib=rb-4.1.0","ciudad":"Mirador del Cap de la Barra, Costa Brava"},
    {"url":"https://images.unsplash.com/photo-1677939217436-01d7c0b8738e?q=80&w=1740&auto=format&fit=crop&ixlib=rb-4.1.0","ciudad":"Torre del Oro, Sevilla"},
    {"url":"https://cdn.pixabay.com/photo/2020/05/08/22/51/national-park-5147616_1280.jpg","ciudad":"Aigüestortes, Lleida"},
    {"url":"https://images.unsplash.com/photo-1665157809094-02fc338305f5?q=80&w=2064&auto=format&fit=crop&ixlib=rb-4.1.0","ciudad":"Alt Pirineu, Lleida"},
    {"url":"https://multimedia.comunitatvalenciana.com/B5B34B4AEFC64B248A719A3B64306FD9/img/E991F48A7CEC482AA06F299319096C07/costa_de_azahar.jpg?responsive","ciudad":"Costa Azahar, Comunitat Valenciana"},
    {"url":"https://images.unsplash.com/photo-1677939217436-01d7c0b8738e?q=80&w=1740&auto=format&fit=crop&ixlib=rb-4.1.0","ciudad":"Torre del Oro, Sevilla"},
    {"url":"https://images.unsplash.com/photo-1677939217436-01d7c0b8738e?q=80&w=1740&auto=format&fit=crop&ixlib=rb-4.1.0","ciudad":"Torre del Oro, Sevilla"},
    {"url":"https://images.unsplash.com/photo-1677939217436-01d7c0b8738e?q=80&w=1740&auto=format&fit=crop&ixlib=rb-4.1.0","ciudad":"Torre del Oro, Sevilla"}
]

# =========================
# SESSION STATE
# =========================
if "imagen_idx" not in st.session_state:
    st.session_state.imagen_idx = 0
if "seccion" not in st.session_state:
    st.session_state.seccion = "Inicio"

def avanzar(): st.session_state.imagen_idx = (st.session_state.imagen_idx + 1) % len(imagenes)
def retroceder(): st.session_state.imagen_idx = (st.session_state.imagen_idx - 1) % len(imagenes)

# =========================
# MENU SUPERIOR
# =========================
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

opcion = st.session_state.seccion

# =========================
# SECCIONES
# =========================
if opcion == "Inicio":
    # Mostrar imagen hero
    imagen_actual = imagenes[st.session_state.imagen_idx]
    st.markdown(f"""
        <div class="img-wrapper">
            <img src="{imagen_actual['url']}" alt="{imagen_actual['ciudad']}">
            <div class="city-label">{imagen_actual['ciudad']}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 30, 1])
    if col1.button('◀'): retroceder()
    if col3.button('▶'): avanzar()

    # Contenido
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

elif opcion == "Seleccionar destino alternativo":
    st.subheader("🔍 Recomendador de destinos turísticos alternativos")
    st.info("Aquí podrás introducir tu destino actual y ver sugerencias menos saturadas con características similares.")
    # TODO: lógica de recomendación

elif opcion == "Ver mapas de saturación":
    st.subheader("Mapa de saturación turística por zona")
    st.info("Visualiza la concentración de turistas en cada zona mediante columnas proporcionales al número de viajeros.")

    # Filtros
    st.markdown("### 🎚️ Filtros temporales y tipo de turismo")

    col_f1, col_f2, col_f3 = st.columns([1.2, 1.2, 2])

    with col_f1:
        años_disponibles = sorted(df["AÑO"].dropna().unique())
        opciones_año = ["Todos los años"] + [str(a) for a in años_disponibles]
        año_seleccionado = st.selectbox("📅 Año", opciones_año)

    with col_f2:
        meses_nombres = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
        meses_disponibles = sorted(df["MES"].dropna().unique())
        opciones_mes = ["Todos los meses"] + [meses_nombres[m] for m in meses_disponibles]
        mes_seleccionado = st.selectbox("🗓️ Mes", opciones_mes)

    with col_f3:
        tipo_seleccionado = st.multiselect(
            "🏨 Tipo de turismo",
            ["Turismo Hotelero", "Turismo Rural", "Apartamentos", "Campings"],
            default=["Turismo Hotelero"]
        )

    columnas_tipo = {
        "Turismo Hotelero": "VIAJEROS_EOH",
        "Turismo Rural": "VIAJEROS_EOTR",
        "Apartamentos": "VIAJEROS_EOAP",
        "Campings": "VIAJEROS_EOAC"
    }
    columnas_seleccionadas = [columnas_tipo[t] for t in tipo_seleccionado] if tipo_seleccionado else []

    # Aplicar filtros
    df_filtrado = df.copy()
    if año_seleccionado != "Todos los años":
        df_filtrado = df_filtrado[df_filtrado["AÑO"] == int(año_seleccionado)]
    if mes_seleccionado != "Todos los meses":
        mes_num = [k for k, v in meses_nombres.items() if v == mes_seleccionado][0]
        df_filtrado = df_filtrado[df_filtrado["MES"] == mes_num]

    # Columna acumulada de viajeros
    if columnas_seleccionadas:
        df_filtrado["viajeros"] = df_filtrado[columnas_seleccionadas].sum(axis=1)
    else:
        df_filtrado["viajeros"] = 0

    # Agrupar por zona y coords
    df_grouped = df_filtrado.groupby(["ZONA_TURISTICA", "lat", "long"], as_index=False)["viajeros"].sum()

    # Colores del mapa / tooltip
    fill_rgba = hex_to_rgba(COLORS["indigo_dye"], alpha=0.70)
    tooltip_bg = COLORS["anti_flash_white"]
    tooltip_border = "#a3bfd2"  # paynes gray 800
    tooltip_text = COLORS["indigo_dye"]

    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=39, longitude=-3.5, zoom=3.5, pitch=30),
        layers=[
            pdk.Layer(
                "ColumnLayer",
                data=df_grouped,
                get_position='[long, lat]',
                get_elevation='viajeros',
                elevation_scale=0.1,
                radius=10000,
                get_fill_color=fill_rgba,
                pickable=True,
                auto_highlight=True,
            )
        ],
        tooltip={
            "html": f"""
                <div style='font-family: Segoe UI; font-size: 13px;'>
                    <strong style='color:{tooltip_text};'>{{ZONA_TURISTICA}}</strong><br>
                    <span style='color:{tooltip_text};'>Viajeros: {{viajeros:,.0f}}</span>
                </div>
            """,
            "style": {
                "backgroundColor": tooltip_bg,
                "color": tooltip_text,
                "border": f"1px solid {tooltip_border}",
                "borderRadius": "8px",
                "padding": "10px"
            }
        }
    ))

elif opcion == "Consultar datos históricos":
    st.subheader("📈 Evolución de datos turísticos")
    st.info("Consulta viajeros, pernoctaciones y ocupación a lo largo del tiempo.")
    # TODO: gráficos y tablas

elif opcion == "Acerca del proyecto":
    st.subheader("ℹ️ Acerca de este proyecto")
    st.write("""
Este proyecto forma parte del Trabajo de Fin de Máster (TFM) en el que se desarrolla un sistema
de recomendación y análisis para mejorar la distribución del turismo en España, utilizando
herramientas de ciencia de datos, visualización interactiva y modelado predictivo.
    """)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("<p class='footer-note'>© 2025 Proyecto TFM - Redistribución Turística | Deep5</p>", unsafe_allow_html=True)
