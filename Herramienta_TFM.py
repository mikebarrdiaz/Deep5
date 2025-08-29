import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from streamlit_autorefresh import st_autorefresh
import altair as alt
import os, glob


# =========================
# CONFIG BASICA DE LA APP
# =========================
st.set_page_config(
    page_title="RedisTour",
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


st.markdown("""
<style>
/* Hero */
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
  box-shadow: 0 4px 14px rgba(0,0,0,0.18);
}
.city-label {
  position: absolute;
  bottom: 20px;
  right: 30px;
  background: rgba(0,0,0,0.45);
  color: #fff;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  backdrop-filter: blur(3px);
}

/* Header: solo tamaños y espaciado (sin colores) */
.header-container {
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.header-container h1 {
  margin: 0;
  font-size: 3rem;
}
.header-container p {
  margin: 0;
  font-size: 1.1rem;
}

/* Footer: solo aspecto neutro */
.footer-note {
  text-align: center;
  font-size: 0.9rem;
  opacity: .9;
}
</style>
""", unsafe_allow_html=True)


# =========================
# CARGA DE DATOS (mapa)
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
# RECOMENDADOR k-NN (Destino alternativo)
# =========================
@st.cache_data(show_spinner=False)
def cargar_datazt():
    df_zt = pd.read_excel("./Data_Dataestur/DataZT.xlsx", sheet_name="Data ZT")
    # Forzar string en categóricas por si hay NaNs/mixtos
    for col in [
        "Tipo_Ubicación", "Clima_Köppen", "Tipo_Turismo_Principal",
        "Estacionalidad_Climática", "Nivel_Infraestructura_Turística",
        "Actividad principal 1", "Actividad principal 2"
    ]:
        if col in df_zt.columns:
            df_zt[col] = df_zt[col].astype(str)
    return df_zt

@st.cache_resource(show_spinner=False)
def entrenar_pipeline(df_zt: pd.DataFrame):
    features = [
        'Tipo_Ubicación',
        'Clima_Köppen',
        'Altitud_Media_msnm',
        'Tipo_Turismo_Principal',
        'Estacionalidad_Climática',
        'Nivel_Infraestructura_Turística',
        'Actividad principal 1',
        'Actividad principal 2'
    ]
    # Subconjunto para el modelo
    df_knn = df_zt[features].copy()

    # Columnas
    categorical_cols = df_knn.select_dtypes(include='object').columns.tolist()
    numerical_cols = ['Altitud_Media_msnm']

    # Preprocesado
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ])

    # k-NN
    knn_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('knn', NearestNeighbors(n_neighbors=10, metric='euclidean'))
    ])
    knn_pipeline.fit(df_knn)
    return knn_pipeline, df_knn, features


# =========================
# ENCABEZADO (logos + texto)
# =========================
col_logo, col_text = st.columns([2, 9])

with col_logo:
    st.image("./Logos/Redisstour.svg", width=250)

with col_text:
    st.markdown("""
        <div class="header-container">
            <h1>Plataforma de Redistribución Inteligente del Turismo</h1>
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
    {"url":"https://cdn.pixabay.com/photo/2022/11/18/16/53/spain-7600551_1280.jpg","ciudad":"Los Alcornocales, Cádiz"},
    {"url":"https://mediaim.expedia.com/destination/1/f8e3b5569445fd06122bf4f0bbee0806.jpg","ciudad":"Cadí-Moixeró, Barcelona"},
    {"url":"https://www.barcelo.com/guia-turismo/wp-content/uploads/ok-costa-vizcaina.jpg","ciudad":"Costa Bizkaia, Pais Vasco"},
    {"url":"https://www.andaluciasimple.com/wp-content/uploads/2020/11/AdobeStock_132882206-scaled.jpeg","ciudad":"Costa Del Sol (Málaga), Andalucía"}
]

# =========================
# SESSION STATE
# =========================
if "imagen_idx" not in st.session_state:
    st.session_state.imagen_idx = 0
if "seccion" not in st.session_state:
    st.session_state.seccion = "Inicio"
# Para controlar el avance automático y evitar dobles incrementos
if "last_refresh_count" not in st.session_state:
    st.session_state.last_refresh_count = -1

def avanzar(): st.session_state.imagen_idx = (st.session_state.imagen_idx + 1) % len(imagenes)
def retroceder(): st.session_state.imagen_idx = (st.session_state.imagen_idx - 1) % len(imagenes)

# =========================
# MENU SUPERIOR
# =========================
col_left, col0, col1, col2, col3, col4, col5, col_right = st.columns([1, 2, 2, 2, 2, 2, 2, 1])

with col0:
    if st.button("Inicio", use_container_width=True):
        st.session_state.seccion = "Inicio"
with col1:
    if st.button("Destino alternativo", use_container_width=True):
        st.session_state.seccion = "Seleccionar destino alternativo"
with col2:
    if st.button("Mapa saturación", use_container_width=True):
        st.session_state.seccion = "Ver mapas de saturación"
with col4:
    if st.button("Datos históricos", use_container_width=True):
        st.session_state.seccion = "Consultar datos históricos"
with col5:
    if st.button("Acerca del proyecto", use_container_width=True):
        st.session_state.seccion = "Acerca del proyecto"
with col3:
    if st.button("Encuentra tu destino", use_container_width=True):
        st.session_state.seccion = "Encuentra tu destino"

opcion = st.session_state.seccion


# =========================
# SECCIONES
# =========================
if opcion == "Inicio":
    # ---- Auto-refresco cada 2s y avance automático ----
    refresh_count = st_autorefresh(interval=4000, key="auto_refresh_hero")

    # Avanzar automático solo si no hay pulsación manual
    if refresh_count != st.session_state.last_refresh_count:
        avanzar()
        st.session_state.last_refresh_count = refresh_count

    # Mostrar imagen hero
    imagen_actual = imagenes[st.session_state.imagen_idx]
    st.markdown(f"""
        <div class="img-wrapper">
            <img src="{imagen_actual['url']}" alt="{imagen_actual['ciudad']}">
            <div class="city-label">{imagen_actual['ciudad']}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Botones debajo de la imagen ----
    col1, col2, col3 = st.columns([1, 30, 1])
    if col1.button('◀'):
        retroceder()
    if col3.button('▶'):
        avanzar()


    st.markdown("<br>", unsafe_allow_html=True)

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
    st.info("Selecciona el tipo de estancia, el mes y el año; luego lanza la búsqueda para ver un único ranking por ocupación con tus zonas similares (incluida la seleccionada).")

    # ========= Carga datos base =========
    df_zt = cargar_datazt()
    if "Nombre_Zona" not in df_zt.columns:
        st.error("⚠️ Error: falta la columna 'Nombre_Zona' en los datos de zonas.")
    else:
        # Entrenar pipeline KNN (cacheado en tu función)
        knn_pipeline, df_knn, features = entrenar_pipeline(df_zt)
        zona_nombres = df_zt['Nombre_Zona'].astype(str).tolist()


        @st.cache_data
        def cargar_forecasts():
            try:
                df_f = pd.read_excel("./Forecasts_2025_2026.xlsx")
            except Exception as e:
                return None, f"No se pudo leer el Excel de forecasts: {e}"
            # Tipos
            for c in ["AÑO", "MES"]:
                if c in df_f.columns:
                    df_f[c] = pd.to_numeric(df_f[c], errors="coerce").astype("Int64")
            return df_f, None

        df_fore, err_fore = cargar_forecasts()
        if err_fore:
            st.error(err_fore)

        # ========= Controles previos (tipo, métrica, fecha y zona) =========
        # Meses en español (sin número)
        MESES_ES = [
            "Enero","Febrero","Marzo","Abril","Mayo","Junio",
            "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"
        ]

        TIPO_TO_COLS = {
            "Turismo rural": {
                "Plazas": "GRADO_OCUPA_PLAZAS_EOTR",
                "Plazas Fin de Semana": "GRADO_OCUPA_PLAZAS_FIN_SEMANA_EOTR",
                "Habitaciones": "GRADO_OCUPA_HABITACIONES_EOTR",
            },
            "Hotel": {
                "Plazas": "GRADO_OCUPA_PLAZAS_EOH",
                "Plazas Fin de Semana": "GRADO_OCUPA_PLAZAS_FIN_SEMANA_EOH",
                "Habitaciones": "GRADO_OCUPA_POR_HABITACIONES_EOH",
            },
            "Apartamentos": {
                "Plazas": "GRADO_OCUPA_PLAZAS_EOAP",
                "Apartamentos": "GRADO_OCUPA_APART_EOAP",
                "Apartamentos Fin de Semana": "GRADO_OCUPA_APART_FIN_SEMANA_EOAP",
            },
            "Camping": {
                "Parcelas": "GRADO_OCUPA_PARCELAS_EOAC",
                "Parcelas Fin de Semana": "GRADO_OCUPA_PARCELAS_FIN_SEMANA_EOAC",
            },
        }


        colA, colB, colC, colD, colE = st.columns([2, 2, 1.7, 1, 1])
        with colA:
            zona_objetivo = st.selectbox(
                "Destino actual",
                options=zona_nombres,
                index=0,
                help="Zona desde la que quieres buscar alternativas similares."
            )
        with colB:
            tipo_estancia = st.selectbox("Tipo de estancia", list(TIPO_TO_COLS.keys()))

        with colC:
            # Mostramos los nombres bonitos
            metrica_bonita = st.selectbox("Métrica de ocupación", list(TIPO_TO_COLS[tipo_estancia].keys()))
            # Recuperamos el nombre real de la columna
            colname = TIPO_TO_COLS[tipo_estancia][metrica_bonita]

            
        with colD:
            año_sel = st.selectbox("Año", [2025, 2026], index=0)
        
        with colE:
            mes_nombre = st.selectbox("Mes", MESES_ES, index=0)
            mes_sel = MESES_ES.index(mes_nombre) + 1      

        colE, colF = st.columns([1, 1])
        with colE:
            k_recom = st.slider("N.º de sugerencias similares", min_value=3, max_value=12, value=5, step=1)
        with colF:
            buscar = st.button("🔎 Buscar destinos alternativos", use_container_width=True)

        # ========= Acción: buscar similares y mostrar ÚNICA tabla =========
        if buscar:
            if df_fore is None:
                st.error("No hay forecasts disponibles; no se puede generar el ranking. Genera primero el Excel.")
            else:
                # 1) KNN similares
                try:
                    indice_zona = zona_nombres.index(zona_objetivo)
                except ValueError:
                    st.error("No se encontró la zona seleccionada en los datos.")
                else:
                    zona_vector = df_knn.iloc[[indice_zona]]
                    distancias, indices = knn_pipeline.named_steps['knn'].kneighbors(
                        knn_pipeline.named_steps['preprocessor'].transform(zona_vector),
                        n_neighbors=k_recom + 1  # +1 incluye la propia
                    )

                    # Construimos similitudes (0–100) y añadimos SIEMPRE la seleccionada (similitud 100)
                    similares = []
                    for j, i in enumerate(indices[0]):
                        nombre = zona_nombres[i]
                        dist = float(distancias[0][j])
                        similares.append({"Zona": nombre, "Distancia": dist})

                    # Quitamos duplicados preservando orden
                    seen = set()
                    filtrados = []
                    for row in similares:
                        if row["Zona"] in seen:
                            continue
                        seen.add(row["Zona"])
                        filtrados.append(row)

                    # Calculamos similitud (0–100) relativa al p95 de las distancias (evita outliers)
                    dists = [r["Distancia"] for r in filtrados if r["Zona"] != zona_objetivo]
                    if len(dists) == 0:
                        p95 = 1.0
                    else:
                        p95 = np.percentile(dists, 95)
                        p95 = p95 if p95 > 0 else 1.0

                    tabla_sim = []
                    for r in filtrados:
                        if r["Zona"] == zona_objetivo:
                            sim = 100.0
                        else:
                            sim = 100.0 * (1.0 - (r["Distancia"] / p95))
                        tabla_sim.append({"Zona": r["Zona"], "Similitud (0-100)": np.clip(sim, 0, 100)})

                    df_sim = pd.DataFrame(tabla_sim)

                    # 2) Unir con forecasts del mes/año elegido (y ordenar solo por ocupación)
                    if colname not in df_fore.columns:
                        st.warning(f"La columna '{colname}' no está en el archivo de forecasts.")
                        st.stop()

                    df_mes = df_fore[(df_fore["AÑO"] == año_sel) & (df_fore["MES"] == mes_sel)].copy()
                    df_mes = df_mes.rename(columns={"ZONA_TURISTICA": "Zona"})
                    df_mes = df_mes[["Zona", colname]]

                    # Zonas a considerar: las que están en similares (incluida la seleccionada)
                    zonas_universo = df_sim["Zona"].astype(str).tolist()
                    df_mes = df_mes[df_mes["Zona"].astype(str).isin(zonas_universo)]

                    if df_mes.empty:
                        st.warning("No hay datos de forecast para ese mes/año en estas zonas.")
                        st.stop()

                    # Merge similitud + forecast
                    df_final = df_sim.merge(df_mes, on="Zona", how="left")
                    df_final = df_final.rename(columns={colname: "Grado de ocupación (%)"})

                    # Filtrar NaN en ocupación
                    df_final = df_final.dropna(subset=["Grado de ocupación (%)"])

                    # Marcar seleccionada
                    df_final.insert(0, "Seleccionada", df_final["Zona"].eq(zona_objetivo).map({True: "✅", False: ""}))

                    # Ordenar columnas: Seleccionada, Zona, Ocupación, Similitud
                    df_final = df_final[["Zona", "Grado de ocupación (%)", "Similitud (0-100)"]]

                    # Ordenar por ocupación ascendente
                    df_final = df_final.sort_values(
                        by=["Similitud (0-100)","Grado de ocupación (%)"],
                        ascending=[False, True]
                    ).reset_index(drop=True)

                    # Formato porcentaje con 1 decimal
                    df_final["Grado de ocupación (%)"] = df_final["Grado de ocupación (%)"].map(lambda x: f"{x:.1f}%")
                    df_final["Similitud (0-100)"] = df_final["Similitud (0-100)"].map(lambda x: f"{x:.1f}%")

                    # Mostrar tabla única
                    st.success(f"Ranking por **{metrica_bonita}** – {mes_nombre} {año_sel}")
                    st.dataframe(df_final, use_container_width=True, hide_index=True)




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
            default=["Turismo Hotelero", "Turismo Rural", "Apartamentos", "Campings"]
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
    df_grouped = df_grouped[df_grouped["viajeros"] > 0]
    df_grouped["viajeros_fmt"] = df_grouped["viajeros"].apply(lambda x: f"{x:,.0f}".replace(",", "."))

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
                <div style='font-family: Satoshi; font-size: 13px;'>
                    <strong style='color:{tooltip_text};'>{{ZONA_TURISTICA}}</strong><br>
                    <span style='color:{tooltip_text};'>Viajeros: {{viajeros_fmt}}</span>
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


elif opcion == "Encuentra tu destino":
    st.subheader("🧭 Encuentra tu destino")
    st.info("Filtra por características y obtén destinos que cumplan tus criterios o, si no existen, sugerencias similares.")

    # Cargar tabla de trabajo
    df_zt = cargar_datazt()

    # Definir columnas/rasgos a usar en filtros
    features = [
        'Tipo_Ubicación',
        'Clima_Köppen',
        'Altitud_Media_msnm',
        'Tipo_Turismo_Principal',
        'Estacionalidad_Climática',
        'Nivel_Infraestructura_Turística',
        'Actividad principal 1',
        'Actividad principal 2'
    ]

    # Normalizaciones mínimas
    for col in features:
        if col in df_zt.columns and df_zt[col].dtype == object:
            df_zt[col] = df_zt[col].astype(str)
    # Asegurar columna nombre
    nombre_col = 'Nombre_Zona' if 'Nombre_Zona' in df_zt.columns else df_zt.columns[0]

    # ===== UI de filtros =====
    colA, colB, colC = st.columns([1,1,1])

    with colA:
        tipo_ubic = st.multiselect("Tipo de ubicación", sorted(df_zt['Tipo_Ubicación'].dropna().unique().tolist()) if 'Tipo_Ubicación' in df_zt else [], placeholder="Elige una opción")
        clima = st.multiselect("Clima (Köppen)", sorted(df_zt['Clima_Köppen'].dropna().unique().tolist()) if 'Clima_Köppen' in df_zt else [], placeholder="Elige una opción")
        tipo_tur = st.multiselect("Tipo de turismo principal", sorted(df_zt['Tipo_Turismo_Principal'].dropna().unique().tolist()) if 'Tipo_Turismo_Principal' in df_zt else [], placeholder="Elige una opción")

    with colB:
        estac = st.multiselect("Estacionalidad climática", sorted(df_zt['Estacionalidad_Climática'].dropna().unique().tolist()) if 'Estacionalidad_Climática' in df_zt else [], placeholder="Elige una opción")
        infra = st.multiselect("Nivel de infraestructura turística", sorted(df_zt['Nivel_Infraestructura_Turística'].dropna().unique().tolist()) if 'Nivel_Infraestructura_Turística' in df_zt else [], placeholder="Elige una opción")
        act1 = st.multiselect("Actividad principal 1", sorted(df_zt['Actividad principal 1'].dropna().unique().tolist()) if 'Actividad principal 1' in df_zt else [], placeholder="Elige una opción")

    with colC:
        act2 = st.multiselect("Actividad principal 2", sorted(df_zt['Actividad principal 2'].dropna().unique().tolist()) if 'Actividad principal 2' in df_zt else [], placeholder="Elige una opción")
        # Slider de altitud
        if 'Altitud_Media_msnm' in df_zt:
            alt_min = int(pd.to_numeric(df_zt['Altitud_Media_msnm'], errors='coerce').min())
            alt_max = int(pd.to_numeric(df_zt['Altitud_Media_msnm'], errors='coerce').max())
            alt_sel = st.slider("Altitud media (msnm)", min_value=alt_min, max_value=alt_max, value=(alt_min, alt_max), step=max(1,(alt_max-alt_min)//100 or 1))
        else:
            alt_sel = None

    # Opciones
    col_opts1, col_opts2 = st.columns([1,1])
    with col_opts1:
        k_sugerencias = st.slider("N.º de sugerencias similares", 3, 10, 5, 1)
    with col_opts2:
        fallback_similares = st.checkbox("Si no hay coincidencias, sugerir similares", value=True)

    # Botón ejecutar
    if st.button("🔎 Buscar destinos", use_container_width=True):
        # -------- Filtro exacto
        df_fil = df_zt.copy()

        def filtrar_in(df_fil, col, valores):
            if valores and col in df_fil:
                return df_fil[df_fil[col].isin(valores)]
            return df_fil

        # Uso:
        df_fil = filtrar_in(df_fil, "Tipo_Ubicación", tipo_ubic)
        df_fil = filtrar_in(df_fil, "Clima_Köppen", clima)
        df_fil = filtrar_in(df_fil, "Tipo_Turismo_Principal", tipo_tur)
        df_fil = filtrar_in(df_fil, "Estacionalidad_Climática", estac)
        df_fil = filtrar_in(df_fil, "Nivel_Infraestructura_Turística", infra)
        df_fil = filtrar_in(df_fil, "Actividad principal 1", act1)
        df_fil = filtrar_in(df_fil, "Actividad principal 2", act2)


        if alt_sel and 'Altitud_Media_msnm' in df_fil.columns:
            df_fil = df_fil[pd.to_numeric(df_fil['Altitud_Media_msnm'], errors='coerce').between(alt_sel[0], alt_sel[1])]

        if len(df_fil) > 0:
            st.success(f"Se han encontrado {len(df_fil)} destinos que cumplen tus criterios.")
            cols_mostrar = [c for c in [nombre_col] + features if c in df_fil.columns]
            st.dataframe(df_fil[cols_mostrar].sort_values(by=[nombre_col]).reset_index(drop=True), use_container_width=True)
        else:
            if not fallback_similares:
                st.warning("No se han encontrado destinos con esos criterios. Activa la casilla de sugerencias para ver alternativas similares.")
            else:
                # -------- Sugerencias similares (k‑NN)
                st.info("No hubo coincidencias exactas. Mostrando destinos similares a tus preferencias.")
                # Entrenar/obtener pipeline ya definido arriba
                knn_pipeline, df_knn, _ = entrenar_pipeline(df_zt)

                # Construir una fila de consulta a partir de los filtros (toma el 1.º valor si hay varios; si no, usa modo/mediana)
                def pick_value(col, seleccion, default_func):
                    if seleccion:
                        return str(seleccion[0])
                    if col in df_zt.columns:
                        # Valor más frecuente
                        return str(df_zt[col].mode(dropna=True).iloc[0]) if df_zt[col].notna().any() else ""
                    return default_func()

                q = {}
                q['Tipo_Ubicación'] = pick_value('Tipo_Ubicación', tipo_ubic, lambda: "")
                q['Clima_Köppen'] = pick_value('Clima_Köppen', clima, lambda: "")
                q['Tipo_Turismo_Principal'] = pick_value('Tipo_Turismo_Principal', tipo_tur, lambda: "")
                q['Estacionalidad_Climática'] = pick_value('Estacionalidad_Climática', estac, lambda: "")
                q['Nivel_Infraestructura_Turística'] = pick_value('Nivel_Infraestructura_Turística', infra, lambda: "")
                q['Actividad principal 1'] = pick_value('Actividad principal 1', act1, lambda: "")
                q['Actividad principal 2'] = pick_value('Actividad principal 2', act2, lambda: "")
                if 'Altitud_Media_msnm' in df_zt.columns:
                    if alt_sel:
                        q['Altitud_Media_msnm'] = np.mean(alt_sel)
                    else:
                        q['Altitud_Media_msnm'] = float(pd.to_numeric(df_zt['Altitud_Media_msnm'], errors='coerce').median())
                else:
                    q['Altitud_Media_msnm'] = 0.0

                # DataFrame de consulta
                q_df = pd.DataFrame([q])

                # Vecinos más cercanos
                dist, idx = knn_pipeline.named_steps['knn'].kneighbors(
                    knn_pipeline.named_steps['preprocessor'].transform(q_df),
                    n_neighbors=k_sugerencias
                )

                resultados = []
                nombres = df_zt[nombre_col].astype(str).tolist()
                for j, i in enumerate(idx[0]):
                    resultados.append({
                        "Zona sugerida": nombres[i],
                        "Distancia": float(dist[0][j])
                    })

                res_df = pd.DataFrame(resultados)
                # Escala de similitud 0–100 (relativa)
                p95 = np.percentile(res_df["Distancia"], 95) if len(res_df) > 1 else res_df["Distancia"].max()
                p95 = p95 if p95 > 0 else 1.0
                res_df["Similitud (0-100)"] = (100 * (1 - res_df["Distancia"]/p95)).clip(0, 100).round(1)

                st.success("Sugerencias similares")
                st.dataframe(res_df.sort_values(["Distancia","Zona sugerida"]).reset_index(drop=True), use_container_width=True)

                with st.expander("🎛️ Preferencias usadas para la similitud"):
                    st.write(pd.DataFrame([q]))



elif opcion == "Consultar datos históricos":

    st.subheader("📈 Datos históricos del turismo")
    st.caption("Analiza la evolución temporal por zona turística, tipo de alojamiento y periodo.")

    # ---------- Config & helpers ----------
    meses_nombres = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
    columnas_tipo = {
        "Turismo Hotelero": "VIAJEROS_EOH",
        "Turismo Rural": "VIAJEROS_EOTR",
        "Apartamentos": "VIAJEROS_EOAP",
        "Campings": "VIAJEROS_EOAC"
    }

    # ---------- Filtros (en línea) ----------

    c1, c2, c3 = st.columns([1.5, 2.5, 2.5])

    with c1:
        # Rango de años
        años = sorted(df["AÑO"].dropna().unique())
        if not len(años):
            st.warning("No hay datos de años en el dataset.")
            st.stop()
        año_min, año_max = int(min(años)), int(max(años))
        # "Todos" = rango completo
        año_opciones = ["Todos"] + [str(a) for a in años]
        año_sel = st.selectbox("Años", options=año_opciones, index=0)
        if año_sel == "Todos":
            año_rango = (año_min, año_max)
        else:
            año_sel = int(año_sel)
            año_rango = (año_sel, año_sel)

    with c2:
        # Meses
        meses_nombres = {
            1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
            7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"
        }
        meses_disponibles = sorted(df["MES"].dropna().unique().tolist())
        meses_labels = [meses_nombres.get(m, str(m)) for m in meses_disponibles]
        meses_opciones = ["Todos"] + meses_labels
        mes_sel = st.selectbox("Meses", options=meses_opciones, index=0)
        meses_sel = meses_disponibles if mes_sel == "Todos" else [k for k,v in meses_nombres.items() if v == mes_sel]

    with c3:
        # Tipos de turismo
        tipos_all = list(columnas_tipo.keys())
        tipos_opciones = ["Todos"] + tipos_all
        tipo_sel = st.selectbox("Tipo de alojamiento", options=tipos_opciones, index=0)
        tipos_sel = tipos_all if tipo_sel == "Todos" else [tipo_sel]

    c4, c5 = st.columns([2.5, 3])

    with c4:
        # Zonas
        zonas_all = sorted(df["ZONA_TURISTICA"].dropna().unique().tolist())
        zonas_opciones = ["Todas"] + zonas_all
        zona_sel = st.selectbox("Zona turística", options=zonas_opciones, index=0)
        zonas_sel = zonas_all if zona_sel == "Todas" else [zona_sel]

    with c5:
        # Nivel de agregación temporal
        nivel = st.radio("Agregación temporal", ["Mensual", "Trimestral", "Anual"], horizontal=True)

    # ---------- Preparación de datos ----------
    if not tipos_sel:
        st.warning("Selecciona al menos un tipo de alojamiento.")
        st.stop()

    cols_metric = [columnas_tipo[t] for t in tipos_sel]
    df_h = df.copy()
    df_h = df_h[(df_h["AÑO"].between(año_rango[0], año_rango[1])) & (df_h["MES"].isin(meses_sel))]
    if zonas_sel:
        df_h = df_h[df_h["ZONA_TURISTICA"].isin(zonas_sel)]

    # Suma de viajeros de los tipos seleccionados
    for c in cols_metric:
        if c not in df_h.columns:
            df_h[c] = 0
    df_h["VIAJEROS_SEL"] = df_h[cols_metric].sum(axis=1)

    # Crear fecha y campos útiles
    df_h["FECHA"] = pd.to_datetime(df_h["AÑO"].astype(int).astype(str) + "-" + df_h["MES"].astype(int).astype(str) + "-01")
    df_h["TRIM"] = pd.PeriodIndex(df_h["FECHA"], freq="Q").astype(str)

    # Agregación por nivel
    group_keys = ["ZONA_TURISTICA"]
    if nivel == "Mensual":
        group_keys += ["AÑO", "MES", "FECHA"]
    elif nivel == "Trimestral":
        group_keys += ["AÑO", "TRIM"]
    else:  # Anual
        group_keys += ["AÑO"]

    agg = df_h.groupby(group_keys, as_index=False)["VIAJEROS_SEL"].sum()

    # ---------- KPIs ----------
    # KPI Total periodo filtrado
    total_periodo = int(agg["VIAJEROS_SEL"].sum()) if len(agg) else 0

    # KPI YoY (comparar último año disponible vs anterior, mis mas)
    yoy_txt = "N/D"
    try:
        ult_anio = int(df_h["AÑO"].max())
        ant_anio = ult_anio - 1
        tot_ult = int(df_h.loc[df_h["AÑO"] == ult_anio, "VIAJEROS_SEL"].sum())
        tot_ant = int(df_h.loc[df_h["AÑO"] == ant_anio, "VIAJEROS_SEL"].sum())
        if tot_ant > 0:
            yoy = (tot_ult / tot_ant - 1) * 100
            yoy_txt = f"{yoy:+.1f}%"
        else:
            yoy_txt = "N/D"
    except Exception:
        pass

    # KPI Top zona en el periodo
    top_zona_txt = "N/D"
    if len(df_h):
        top_zona = df_h.groupby("ZONA_TURISTICA", as_index=False)["VIAJEROS_SEL"].sum().sort_values("VIAJEROS_SEL", ascending=False).head(1)
        if len(top_zona):
            top_zona_txt = f"{top_zona.iloc[0]['ZONA_TURISTICA']} ({int(top_zona.iloc[0]['VIAJEROS_SEL']):,}".replace(",", ".") + ")"

    k1, k2, k3 = st.columns([3, 3, 4])
    k1.metric("Viajeros", f"{total_periodo:,}".replace(",", "."))
    k2.metric("Variación YoY (último año vs. anterior)", yoy_txt)
    k3.metric("Zona top", top_zona_txt)

    st.divider()

    # ---------- Series temporales ----------
    st.markdown("#### Evolución temporal")

    if nivel == "Mensual":
        x_field = "FECHA"
        x_title = "Fecha"
    elif nivel == "Trimestral":
        x_field = "TRIM"
        x_title = "Trimestre"
    else:
        x_field = "AÑO"
        x_title = "Año"

    # Serie total (todas las zonas) + por zona (interactiva)
    agg_total = agg.groupby(x_field, as_index=False)["VIAJEROS_SEL"].sum().sort_values(x_field)

    base_total = alt.Chart(agg_total).mark_line(point=True).encode(
        x=alt.X(x_field, title=x_title, sort=None),
        y=alt.Y("VIAJEROS_SEL:Q", title="Viajeros"),
        tooltip=[x_field, alt.Tooltip("VIAJEROS_SEL:Q", title="Viajeros", format=",.0f")]
    ).properties(height=280)

    st.altair_chart(base_total, use_container_width=True)

    # Serie por zona (facet opcional con selector)
    with st.expander("Ver evolución por zona"):
        zonas_plot = agg["ZONA_TURISTICA"].dropna().unique().tolist()
        zonas_sel_plot = st.multiselect("Zonas a mostrar", zonas_plot, default=zonas_plot[:8])
        df_plot = agg[agg["ZONA_TURISTICA"].isin(zonas_sel_plot)].copy()

        if len(df_plot):
            line = alt.Chart(df_plot).mark_line().encode(
                x=alt.X(x_field, title=x_title, sort=None),
                y=alt.Y("VIAJEROS_SEL:Q", title="Viajeros"),
                color=alt.Color("ZONA_TURISTICA:N", title="Zona"),
                tooltip=["ZONA_TURISTICA", x_field, alt.Tooltip("VIAJEROS_SEL:Q", title="Viajeros", format=",.0f")]
            ).properties(height=320)
            st.altair_chart(line, use_container_width=True)
        else:
            st.warning("No hay datos para las zonas seleccionadas.")

    st.divider()

    # ---------- Heatmap Año vs Mes ----------
    st.markdown("#### Calendario (Año × Mes)")
    # Pivot con suma por año-mes (todas las zonas filtradas)
    df_hm = df_h.groupby(["AÑO", "MES"], as_index=False)["VIAJEROS_SEL"].sum()
    if len(df_hm):
        df_hm["Mes"] = df_hm["MES"].map(meses_nombres)
        heat = alt.Chart(df_hm).mark_rect().encode(
            x=alt.X("Mes:N", sort=list(meses_nombres.values()), title="Mes"),
            y=alt.Y("AÑO:O", title="Año"),
            color=alt.Color("VIAJEROS_SEL:Q", title="Viajeros", scale=alt.Scale(scheme="blues")),
            tooltip=[
                alt.Tooltip("AÑO:O", title="Año"),
                alt.Tooltip("Mes:N", title="Mes"),
                alt.Tooltip("VIAJEROS_SEL:Q", title="Viajeros", format=",.0f")
            ]
        ).properties(height=260)
        st.altair_chart(heat, use_container_width=True)
    else:
        st.info("No hay datos para construir el calendario.")

    st.divider()

    # ---------- Top 10 zonas ----------
    st.markdown("#### Top 10 zonas")
    topN = df_h.groupby("ZONA_TURISTICA", as_index=False)["VIAJEROS_SEL"].sum().sort_values("VIAJEROS_SEL", ascending=False).head(10)
    if len(topN):
        barchart = alt.Chart(topN).mark_bar().encode(
            x=alt.X("VIAJEROS_SEL:Q", title="Viajeros"),
            y=alt.Y("ZONA_TURISTICA:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("VIAJEROS_SEL:Q", title="Viajeros", format=",.0f"), "ZONA_TURISTICA:N"]
        ).properties(height=28*len(topN) + 20)
        st.altair_chart(barchart, use_container_width=True)

        # tabla + descarga
        show_tbl = st.checkbox("Mostrar tabla", value=False)
        if show_tbl:
            tmp = topN.copy()
            tmp["Viajeros"] = tmp["VIAJEROS_SEL"].map(lambda x: f"{x:,.0f}".replace(",", "."))
            st.dataframe(tmp[["ZONA_TURISTICA", "Viajeros"]].reset_index(drop=True), use_container_width=True)

        csv = topN.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar Top 10 (CSV)", data=csv, file_name="top10_zonas_periodo.csv", mime="text/csv")
    else:
        st.info("No hay datos para el Top 10 en el periodo seleccionado.")





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

col1, col2, col3, col4, col5 = st.columns([6, 2, 2, 2, 6])  # repartición de columnas

with col2:
    st.image("./Logos/Logo_Deep5.svg", width=160)

with col3:
    st.markdown(
        "<p style='text-align:center; font-size:0.9rem; margin-top:20px;'>En colaboración con</p>",
        unsafe_allow_html=True
    )

with col4:
    st.image("./Logos/Logo_Tui3.svg", width=100)

st.markdown(
    "<p class='footer-note'>© 2025 Proyecto TFM - Redistribución Turística | Deep5</p>",
    unsafe_allow_html=True
)

