# Proyecto RedisTour
# github: https://github.com/mikebarrdiaz/Deep5

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt
from streamlit.components.v1 import html as html_component
import html
import base64

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

from streamlit_autorefresh import st_autorefresh

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

MESES_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}

# === Ocupación por tipo (columnas por defecto para el desglose) ===
OCC_COLS_DEFAULT = {
    "Hotel": "GRADO_OCUPA_PLAZAS_EOH",
    "Turismo rural": "GRADO_OCUPA_PLAZAS_EOTR",
    "Apartamentos": "GRADO_OCUPA_PLAZAS_EOAP",
    "Camping": "GRADO_OCUPA_PARCELAS_EOAC",
}
# Etiquetas cortas para chips
OCC_LABELS = {
    "Hotel": "🏨 Hotel",
    "Turismo rural": "🏡 Rural",
    "Apartamentos": "🏢 Apart.",
    "Camping": "⛺ Camping",
}

def attach_occupancy_breakdown(df_fore: pd.DataFrame, zonas_list: list[str], año_sel: int, mes_sel: int) -> dict[str, dict]:
    """
    Devuelve {zona: {tipo: valor_float_or_None, ...}} usando OCC_COLS_DEFAULT
    """
    out = {z: {} for z in zonas_list}
    if df_fore is None:
        return out

    cols_exist = [c for c in OCC_COLS_DEFAULT.values() if c in df_fore.columns]
    if not cols_exist:
        return out

    df_mes = df_fore[(df_fore["AÑO"] == año_sel) & (df_fore["MES"] == mes_sel)].copy()
    if "ZONA_TURISTICA" not in df_mes.columns:
        return out

    keep_cols = ["ZONA_TURISTICA"] + cols_exist
    df_mes = df_mes[keep_cols]
    df_mes["ZONA_TURISTICA"] = df_mes["ZONA_TURISTICA"].astype(str)

    tmp = df_mes.set_index("ZONA_TURISTICA").to_dict(orient="index")
    for z in zonas_list:
        zstr = str(z)
        if zstr in tmp:
            values = tmp[zstr]
            for tipo, col in OCC_COLS_DEFAULT.items():
                out[z].update({tipo: float(values[col]) if (col in values and pd.notna(values[col])) else None})
        else:
            for tipo in OCC_COLS_DEFAULT.keys():
                out[z].update({tipo: None})
    return out

def hex_to_rgba(hex_str, alpha=1.0):
    hex_str = hex_str.strip("#")
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    a = int(max(0, min(1, alpha)) * 255)
    return [r, g, b, a]

def format_pct(x, nan_txt="-"):
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return nan_txt

def p95_normalized_similarity(distances: list[float]) -> float:
    """Devuelve el percentil 95 (para normalizar similitud), con mínimos de seguridad."""
    if not distances:
        return 1.0
    if len(distances) == 1:
        return max(distances[0], 1e-9)
    p95 = np.percentile(distances, 95)
    return max(float(p95), 1e-9)


# =========================
# RUTAS (robustas)
# =========================
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "Data_Dataestur"
LOGOS_DIR = BASE / "Logos"


# =========================
# ESTILOS (1 bloque)
# =========================
st.markdown("""
<style>
.img-wrapper { position: relative; width: 100%; height: 450px; }
.img-wrapper img { width: 100%; height: 100%; object-fit: cover; border-radius: 5px; box-shadow: 0 4px 14px rgba(0,0,0,0.18); }
.city-label { position: absolute; bottom: 20px; right: 30px; background: rgba(0,0,0,0.45); color: #fff; padding: 0.5rem 1rem; border-radius: 8px; font-size: 1rem; font-weight: 600; backdrop-filter: blur(3px); }
.header-container { display: flex; flex-direction: column; justify-content: center; }
.header-container h1 { margin: 0; font-size: 3rem; }
.header-container p { margin: 0; font-size: 1.1rem; }
.footer-note { text-align: center; font-size: 0.9rem; opacity: .9; }

/* Tarjetas */
.desc-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 18px;
  width: 100%;
}
.desc-card {
  background: #ffffff;
  border: 1px solid #d9e2ea;
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 4px 14px rgba(0,0,0,0.06);
}
.badges { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px; }
.badge {
  background:#e6eaed; color:#224762; border-radius: 999px;
  padding: 2px 10px; font-size: .75rem; font-weight:600;
}
.card-title { font-weight: 800; color:#224762; margin: 4px 0 4px 0; font-size: 1.05rem; }
.meta { color:#4a5a67; font-size:.92rem; margin:0 0 8px 0; }
.kpis { display:flex; gap:12px; flex-wrap:wrap; margin: 6px 0 8px 0; }
.kpi { background:#f5f7f9; border:1px solid #e5eef5; border-radius:10px; padding:6px 10px; font-size:.86rem; color:#224762; }
.desc-body { color: #3a4b59; margin: 0; line-height: 1.35; font-size: 0.95rem; }
.sel { border: 2px solid #306388; }
.section-title { margin: 8px 0 6px 0; font-weight: 800; color:#224762; }
</style>
""", unsafe_allow_html=True)


# =========================
# CARGA DE DATOS (mapa) – Excel
# =========================
REQ_COLS_TOTAL = ["ZONA_TURISTICA", "AÑO", "MES",
                  "VIAJEROS_EOH", "VIAJEROS_EOTR", "VIAJEROS_EOAP", "VIAJEROS_EOAC"]
REQ_COLS_COORDS = ["ZONA_TURISTICA", "lat", "long"]

def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _normalize_zone_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas eliminando acentos, espacios y mayúsculas/minúsculas.
       Mapea a: ZONA_TURISTICA, DESCRIPCION, Comunidad_Autonoma, Provincia
    """
    def deaccent(s: str) -> str:
        return (s.replace("Í","I").replace("í","i")
                 .replace("Ú","U").replace("ú","u")
                 .replace("Á","A").replace("á","a")
                 .replace("É","E").replace("é","e")
                 .replace("Ó","O").replace("ó","o"))
    ren = {}
    for c in df.columns:
        norm = deaccent(c).strip().replace(" ", "_").upper()
        mapping = {
            "ZONA_TURISTICA": "ZONA_TURISTICA",
            "DESCRIPCION": "DESCRIPCION",
            "COMUNIDAD_AUTONOMA": "Comunidad_Autonoma",
            "PROVINCIA": "Provincia",
        }
        if norm in mapping:
            ren[c] = mapping[norm]
    if ren:
        df = df.rename(columns=ren)
    if "ZONA_TURISTICA" in df.columns:
        df["ZONA_TURISTICA"] = df["ZONA_TURISTICA"].astype(str).str.strip()
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def cargar_datos():
    total_excel = DATA_DIR / "DATA_TOTAL.xlsx"
    if not total_excel.exists():
        st.error(f"No se encuentra {total_excel}.")
        st.stop()
    # salt de cache por mtime
    _salt = total_excel.stat().st_mtime
    try:
        df_total = pd.read_excel(total_excel, sheet_name="Total")
        df_coords = pd.read_excel(total_excel, sheet_name="Coordenadas ZT")
    except Exception as e:
        st.error(f"No se pudo abrir el Excel: {e}")
        st.stop()

    df_total = _normalize_zone_colnames(df_total)
    df_coords = _normalize_zone_colnames(df_coords)

    faltan_total = [c for c in REQ_COLS_TOTAL if c not in df_total.columns]
    faltan_coords = [c for c in REQ_COLS_COORDS if c not in df_coords.columns]
    if faltan_total:
        st.error(f"Faltan columnas en DATA_TOTAL: {faltan_total}")
        st.stop()
    if faltan_coords:
        st.error(f"Faltan columnas en COORDS_ZT: {faltan_coords}")
        st.stop()

    df_total = _coerce_numeric(df_total, ["AÑO", "MES", "VIAJEROS_EOH", "VIAJEROS_EOTR", "VIAJEROS_EOAP", "VIAJEROS_EOAC"])

    df_coords = df_coords.drop_duplicates("ZONA_TURISTICA")
    df = df_total.merge(df_coords[["ZONA_TURISTICA", "lat", "long"]], on="ZONA_TURISTICA", how="left")

    return df_total, df_coords, df, _salt

df_total, df_coords, df, _salt_datos = cargar_datos()

# =========================
# DESCRIPCIONES + DATA ZT
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def cargar_descripciones_y_datazt():
    total_excel = DATA_DIR / "DATA_TOTAL.xlsx"
    if not total_excel.exists():
        st.error(f"No se encuentra {total_excel}.")
        st.stop()
    _salt = total_excel.stat().st_mtime

    try:
        df_zt = pd.read_excel(total_excel, sheet_name="Data ZT")
    except Exception as e:
        st.error(f"No se pudo leer la hoja 'Data ZT' en DATA_TOTAL.xlsx: {e}")
        st.stop()
    df_zt = _normalize_zone_colnames(df_zt)

    try:
        df_desc = pd.read_excel(total_excel, sheet_name="Descripciones")
    except Exception as e:
        st.warning(f"No se pudo leer la hoja 'Descripciones': {e}")
        df_desc = pd.DataFrame(columns=["ZONA_TURISTICA", "DESCRIPCION"])

    df_desc = _normalize_zone_colnames(df_desc)
    if "DESCRIPCION" not in df_desc.columns:
        if "Descripcion" in df_desc.columns:
            df_desc = df_desc.rename(columns={"Descripcion": "DESCRIPCION"})
        else:
            df_desc["DESCRIPCION"] = ""

    if "ZONA_TURISTICA" in df_desc.columns:
        df_desc["DESCRIPCION"] = df_desc["DESCRIPCION"].fillna("").astype(str).str.strip()
        df_desc = df_desc.drop_duplicates(subset=["ZONA_TURISTICA"], keep="first")

    return df_zt, df_desc, _salt

df_zt_all, df_desc_all, _salt_desc = cargar_descripciones_y_datazt()

def get_desc_dict(df_desc: pd.DataFrame) -> dict:
    if "ZONA_TURISTICA" not in df_desc.columns or "DESCRIPCION" not in df_desc.columns:
        return {}
    keys = df_desc["ZONA_TURISTICA"].astype(str).str.strip()
    vals = df_desc["DESCRIPCION"].fillna("").astype(str).str.strip()
    return dict(zip(keys, vals))

DESC_MAP = get_desc_dict(df_desc_all)

def _pick_col(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def get_loc_info(zona: str):
    """Intenta devolver (Comunidad, Provincia) desde Data ZT si existen."""
    ca_col = _pick_col(df_zt_all, ["CCAA", "Comunidad Autónoma", "COMUNIDAD_AUTONOMA", "Comunidad_Autonoma"])
    pr_col = _pick_col(df_zt_all, ["Provincia", "PROVINCIA"])
    ca = pr = "—"
    if "ZONA_TURISTICA" in df_zt_all.columns:
        sub = df_zt_all.loc[df_zt_all["ZONA_TURISTICA"].astype(str).str.strip() == str(zona).strip()]
        if len(sub):
            if ca_col: 
                val = sub.iloc[0][ca_col]
                ca = str(val) if pd.notna(val) else "—"
            if pr_col: 
                val = sub.iloc[0][pr_col]
                pr = str(val) if pd.notna(val) else "—"
    return ca, pr


# =========================
# CARGA GLOBAL DE OPINIONES Y FORECASTS 
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def cargar_opiniones_zt():
    total_excel = DATA_DIR / "DATA_TOTAL.xlsx"
    try:
        df_op = pd.read_excel(total_excel, sheet_name="OpinionesZT")
    except Exception as e: 
        return {}, f"No se pudo leer la hoja 'Opiniones ZT': {e}"
    df_op = _normalize_zone_colnames(df_op)
    if "Opiniones" not in df_op.columns:
        for alt_col in ["OPINIONES", "Opinion", "OPINION", "Reseñas", "Resenas"]:
            if alt_col in df_op.columns:
                df_op = df_op.rename(columns={alt_col: "Opiniones"})
                break
    if "ZONA_TURISTICA" not in df_op.columns or "Opiniones" not in df_op.columns:
        return {}, "Faltan columnas en 'Opiniones ZT' (se requieren ZONA_TURISTICA y Opiniones)."
    df_op["ZONA_TURISTICA"] = df_op["ZONA_TURISTICA"].astype(str).str.strip()
    df_op["Opiniones"] = df_op["Opiniones"].fillna("").astype(str).str.strip()
    df_op = df_op[df_op["Opiniones"] != ""]
    op_map = {str(z): sub["Opiniones"].tolist() for z, sub in df_op.groupby("ZONA_TURISTICA")}
    return op_map, None

@st.cache_data(ttl=3600, show_spinner=False)
def cargar_forecasts():
    fpath = BASE / "Forecasts_2025_2026_2027.xlsx"
    try:
        df_f = pd.read_excel(fpath)
    except Exception as e:
        return None, f"No se pudo leer el Excel de forecasts: {e}"
    for c in ["AÑO", "MES"]:
        if c in df_f.columns:
            df_f[c] = pd.to_numeric(df_f[c], errors="coerce").astype("Int64")
    return df_f, None

OPINIONES_MAP, _err_ops_global = cargar_opiniones_zt()
df_fore_global, _err_fore_global = cargar_forecasts()


# =========================
# RECOMENDADOR k-NN (Destino alternativo)
# =========================
@st.cache_resource(show_spinner=False)
def entrenar_pipeline(df_zt: pd.DataFrame):
    features = [
        # Categóricas
        "Tipo_Ubicación",
        "Clima_Köppen",
        "Estacionalidad_Climática",
        "Nivel_Infraestructura_Turística",
        "Aeropuerto_mas_cercano",
        "Tipo_Turismo_Principal",
        "Actividad principal 1",
        "Actividad principal 2",
        "Tipo_entorno_protegido",
        "Patrimonio_cultural",
        "Oferta_complementaria",

        # Numéricas continuas
        "Altitud_Media_msnm",
        "Distancia_al_mar_km",
        "Indice_conectividad",
        "Distancia_aeropuerto_km",
        "Distancia_estacion_tren_km",
        "Porcentaje_area_protegida",

        # Binarias 0/1
        "Actividad_Naturaleza",
        "Actividad_Historico",
        "Actividad_Entretenimiento",
        "Actividad_Montanismo",
        "Actividad_Deportes_Acuaticos",
        "Actividad_Gastronomia",
        "Actividad_Cultural",
        "Actividad_Ocio",
        "Actividad_Senderismo",
        "Actividad_Turismo_rural",
        "Actividad_Astronomia",
        "Actividad_Deportes_de_Invierno",
        "Actividad_Observacion_de_Fauna",
        "Actividad_Playa",
        "Actividad_Cicloturismo",
        "Actividad_Wellness_Termalismo",
        "Actividad_Compras",
        "Actividad_Enoturismo",
        "Actividad_Negocios_MICE",
        "Actividad_Religioso",
        "Actividad_Aventura",
        "Actividad_Turismo_Nautico",
    ]
    feats = [f for f in features if f in df_zt.columns]
    df_knn = df_zt[feats].copy()

    categorical_cols = df_knn.select_dtypes(include='object').columns.tolist()
    numerical_cols = [c for c in df_knn.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ])

    knn_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('knn', NearestNeighbors(n_neighbors=10, metric='cosine'))
    ])
    knn_pipeline.fit(df_knn)
    return knn_pipeline, df_knn, feats

# =========================
# ENCABEZADO (logos + texto)
# =========================
col_logo, col_text = st.columns([2, 9])
with col_logo:
    logo_path = LOGOS_DIR / "Redisstour2.svg"
    if logo_path.exists():
        st.image(str(logo_path), width=250)
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
    {"url":"https://www.barcelo.com/guia-turismo/wp-content/uploads/ok-costa-vizcaina.jpg","ciudad":"Costa Bizkaia, País Vasco"},
    {"url":"https://www.andaluciasimple.com/wp-content/uploads/2020/11/AdobeStock_132882206-scaled.jpeg","ciudad":"Costa del Sol (Málaga), Andalucía"}
]

# =========================
# SESSION STATE
# =========================
if "imagen_idx" not in st.session_state:
    st.session_state.imagen_idx = 0
if "seccion" not in st.session_state:
    st.session_state.seccion = "Inicio"
if "last_refresh_count" not in st.session_state:
    st.session_state.last_refresh_count = -1

def avanzar(): st.session_state.imagen_idx = (st.session_state.imagen_idx + 1) % len(imagenes)
def retroceder(): st.session_state.imagen_idx = (st.session_state.imagen_idx - 1) % len(imagenes)

# =========================
# MENU SUPERIOR (mantenido)
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
    refresh_count = st_autorefresh(interval=4000, key="auto_refresh_hero")
    if refresh_count != st.session_state.last_refresh_count:
        avanzar()
        st.session_state.last_refresh_count = refresh_count

    imagen_actual = imagenes[st.session_state.imagen_idx]
    st.markdown(f"""
        <div class="img-wrapper">
            <img src="{imagen_actual['url']}" alt="{html.escape(imagen_actual['ciudad'])}">
            <div class="city-label">{imagen_actual['ciudad']}</div>
        </div>
    """, unsafe_allow_html=True)

    # Controles hero
    col1b, col2b, col3b = st.columns([1, 30, 1])
    with col1b:
        st.button("◀", key="hero_prev", use_container_width=True, on_click=retroceder, help="Imagen anterior")
    with col3b:
        st.button("▶", key="hero_next", use_container_width=True, on_click=avanzar, help="Imagen siguiente")

    # Estilos profesionales
    st.markdown("""
    <style>
      .pro-wrap{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:12px;}
      @media (max-width: 1100px){ .pro-wrap{grid-template-columns:1fr;} }
      .pro-card{
        background:#fff; border:1px solid #e5eef5; border-radius:14px;
        padding:20px 22px; box-shadow:0 2px 10px rgba(0,0,0,0.05);
      }
      .pro-title{
        margin:0 0 10px 0; font-weight:800; font-size:1.25rem; color:#224762;
      }
      .pro-text{ color:#31414f; font-size:.96rem; line-height:1.55; margin:0; }
      .pro-sub{ color:#506273; font-size:.94rem; line-height:1.5; margin:.4rem 0 0 0; }
      .pro-list{ margin:.5rem 0 0 0; padding-left:0; list-style:none; }
      .pro-list li{
        position:relative; padding-left:18px; margin:.35rem 0; color:#31414f; font-size:.95rem;
      }
      .pro-list li::before{
        content:""; position:absolute; left:0; top:.6rem; width:8px; height:8px;
        border-radius:50%; background:#306388; opacity:.85;
      }
      .pro-accent{
        border-left:3px solid #306388; padding-left:16px; margin-top:8px;
      }
      .pro-kpis{display:flex; gap:10px; flex-wrap:wrap; margin-top:10px;}
      .pro-kpi{
        background:#f7fafc; border:1px solid #e5eef5; color:#224762;
        padding:6px 10px; border-radius:10px; font-size:.85rem; font-weight:600;
      }
    </style>
    """, unsafe_allow_html=True)

    # Contenido en dos columnas
    st.markdown("""
    <div class="pro-wrap">
      <div class="pro-card">
        <h3 class="pro-title">Motivación del proyecto</h3>
        <p class="pro-text pro-accent" style= "margin-bottom:20px;">
          España afronta el reto de gestionar el crecimiento turístico garantizando la sostenibilidad
          y la calidad de vida de sus residentes, sin mermar la experiencia del visitante.
        </p>
        <ul class="pro-list">
          <li>Saturación de destinos maduros en picos estacionales.</li>
          <li>Presión sobre infraestructuras, servicios y recursos locales.</li>
          <li>Impacto ambiental y degradación de la experiencia global.</li>
        </ul>
        <p class="pro-sub">
          Al mismo tiempo, existen múltiples destinos con alto potencial que permanecen infrautilizados.
          La redistribución inteligente del flujo turístico es clave para equilibrar el sistema.
        </p>
      </div>

      <div class="pro-card">
        <h3 class="pro-title">Objetivo del sistema</h3>
        <p class="pro-text pro-accent" style= "margin-bottom:20px;">
          Proveer una plataforma de análisis y apoyo a la decisión que permita
          redistribuir la demanda turística de forma sostenible, informada y eficiente.
        </p>
        <ul class="pro-list">
          <li>Identificación de zonas con riesgo de saturación a partir de históricos.</li>
          <li>Recomendación de destinos alternativos de perfil comparable y menor presión.</li>
          <li>Visualización clara de tendencias y patrones mediante mapas y gráficos.</li>
          <li>Soporte a políticas públicas y estrategia sectorial orientadas a la sostenibilidad.</li>
        </ul>
        <div class="pro-kpis">
          <div class="pro-kpi">Análisis histórico</div>
          <div class="pro-kpi">Recomendación</div>
          <div class="pro-kpi">Visualización</div>
          <div class="pro-kpi">Sostenibilidad</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Extensión: Datos, buen uso y límites ---
    st.markdown("""
    <div class="pro-card" style="margin-top:22px; margin-bottom:22px;">
    <h3 class="pro-title">La importancia de los datos</h3>
    <p class="pro-text pro-accent" style= "margin-bottom:20px;">
        La correcta gestión de la información es un factor decisivo para avanzar hacia un turismo más
        equilibrado y sostenible. Los datos permiten entender la realidad, anticipar riesgos y
        orientar las decisiones estratégicas.
    </p>
    <p class="pro-text">
        La calidad de los resultados depende directamente de la calidad de los datos utilizados.
        Su recopilación, actualización y análisis continuado son esenciales para reflejar de forma
        fiel la dinámica turística en España. Un uso responsable de esta información no solo favorece
        la eficiencia en la planificación, sino que también garantiza transparencia y confianza en
        el proceso de toma de decisiones.
    </p>
    <p class="pro-text">
        Al poner los datos en el centro, esta plataforma busca ofrecer una base sólida sobre la que
        construir políticas públicas y estrategias privadas que promuevan un turismo más justo,
        resiliente y respetuoso con el entorno.
    </p>
    <p class="pro-sub" style="margin-top:14px; font-style:italic; color:#224762;">
        “Los datos son la brújula que guía el futuro del turismo sostenible.”
    </p>
    </div>
    """, unsafe_allow_html=True)

elif opcion == "Seleccionar destino alternativo":
    st.subheader("🔍 Recomendador de destinos turísticos alternativos")
    st.info("Elige mes y año; verás similitud y el % de ocupación desglosado (Hotel, Rural, Aptos, Camping).")

    if _err_ops_global:
        st.warning(_err_ops_global)

    df_zt = df_zt_all.copy()
    if "ZONA_TURISTICA" not in df_zt.columns:
        st.error("⚠️ Falta la columna 'ZONA_TURISTICA' en los datos de zonas.")
    else:
        knn_pipeline, df_knn, features = entrenar_pipeline(df_zt)
        zona_nombres = df_zt['ZONA_TURISTICA'].astype(str).tolist()

        df_fore, err_fore = df_fore_global, _err_fore_global
        if err_fore:
            st.error(err_fore)

        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])

        with c1:
            zona_objetivo = st.selectbox(
                "Destino actual",
                options=zona_nombres,
                index=0,
                help="Zona desde la que quieres buscar alternativas similares."
            )
        with c2:
            año_sel = st.selectbox("Año", [2025, 2026, 2027], index=0)
        with c3:
            mes_nombre = st.selectbox("Mes", list(MESES_ES.values()), index=0)
            mes_sel = [k for k, v in MESES_ES.items() if v == mes_nombre][0]
        with c4:
            k_recom = st.slider("N.º recomendaciones", min_value=3, max_value=12, value=6, step=1)

        buscar = st.button("🔎 Buscar", use_container_width=True)
        if not buscar:
            st.markdown(f"""
                    <div class='desc-card sel' style='margin-top:6px;'>
                        <div class='badges'><span class='badge'>Descripción</span></div>
                        <div class='card-title' style='margin-bottom:4px;'>{zona_objetivo}</div>
                        <p class='desc-body'>{DESC_MAP.get(zona_objetivo, "Sin descripción disponible.")}</p>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        def render_zone_result_cards(rows: list[dict], subtitle: str = "", per_row: int = 1):
            if not rows:
                return

            colA, colB, colC = st.columns([9, 13, 3])
            with colA:
                st.markdown(f"#### {subtitle}")

            with colC:
                logo_path = LOGOS_DIR / "TripAdvisor_Logo.svg"
                if logo_path.exists():
                    st.markdown(
                        "<div style='text-align:left;font-size:0.85rem;color:#6b7280;'>Powered by</div>",
                        unsafe_allow_html=True
                    )
                    st.image(str(logo_path), width=120)

            css_block = """
            <style>
            .desc-card{
              background:#fff;border:1px solid #d9e2ea;border-radius:16px;
              padding:14px 16px;box-shadow:0 4px 14px rgba(0,0,0,0.06);
              font-family:Satoshi,system-ui;overflow:visible;margin-bottom:10px;
            }
            .sel{border:2px solid #306388;}
            .header-row{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:6px;}
            .title-block{display:flex;flex-direction:column;min-width:0;}
            .badges{display:flex;gap:8px;flex-wrap:wrap;margin:0;}
            .badge{background:#e6eaed;color:#224762;border-radius:999px;padding:2px 10px;font-size:.75rem;font-weight:600;}
            .card-title{font-weight:800;color:#224762;margin:0 0 2px 0;font-size:1.05rem;}
            .meta{color:#4a5a67;font-size:.92rem;margin:0;}
            .kpis-head{font-weight:700;color:#224762;font-size:.88rem;margin:4px 0 6px 0;}
            .kpis{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 8px 0;}
            .kpi{background:#f5f7f9;border:1px solid #e5eef5;border-radius:10px;padding:6px 10px;font-size:.86rem;color:#224762;}
            .kpi2{background:#e9eff3;border:1px solid #e5eef5;border-radius:3px;padding:6px 10px;font-size:.86rem;color:#224762;}
            .desc-body{color:#3a4b59;margin:0;line-height:1.35;font-size:0.95rem;}
            .reviews-wrap { margin-top:8px; }
            .reviews-wrap details { background:#f7f9fb; border:1px solid #e5eef5; border-radius:10px; padding:8px 10px; }
            .reviews-wrap summary { cursor:pointer; color:#224762; font-weight:700; }
            .reviews-scroll{ max-height:500px; overflow:auto; margin-top:8px; padding-right:6px; }
            .reviews-list { margin:0 0 0 16px; color:#3a4b59; font-size:0.92rem; }
            .reviews-list li { margin-bottom:6px; }
            </style>
            """

            sel_rows = [r for r in rows if r.get("seleccionada")]
            other_rows = [r for r in rows if not r.get("seleccionada")]
            ordered = sel_rows + other_rows

            def _is_num(x):
                try:
                    return (x is not None) and (not (isinstance(x, float) and np.isnan(x)))
                except Exception:
                    return False

            def kpi_html(r: dict) -> str:
                oc = r.get("ocups")
                if not isinstance(oc, dict) or not len(oc):
                    return ""
                parts = []
                for tipo in [t for t in ["Hotel", "Turismo rural", "Apartamentos", "Camping"] if t in oc]:
                    if _is_num(oc.get(tipo)):
                        parts.append(f"<div class='kpi'>{OCC_LABELS[tipo]}: <b>{format_pct(oc[tipo])}</b></div>")
                return "".join(parts)

            def reviews_html(r: dict) -> str:
                opiniones = r.get("opiniones") or []
                if not opiniones:
                    return ""
                try:
                    n = min(5, len(opiniones))
                    idx = np.random.choice(len(opiniones), size=n, replace=False)
                    sample = [str(opiniones[i]) for i in idx]
                except Exception:
                    sample = [str(op) for op in opiniones[:5]]

                items = "\n".join([f"<li>{html.escape(s)}</li>" for s in sample])
                return f"""
                    <div class='reviews-wrap'>
                        <details>
                            <summary>🌟 Opiniones de viajeros</summary>
                            <div class='reviews-scroll'>
                                <ul class='reviews-list'>
                                    {items}
                                </ul>
                            </div>
                        </details>
                    </div>
                """

            def card_html(r: dict) -> str:
                css_sel = " sel" if r.get("seleccionada") else ""
                zona = r.get("zona", "—")
                ca = r.get("comunidad", "—")
                pr = r.get("provincia", "—")
                si = r.get("similitud") or "No hay datos"
                desc = r.get("desc", "Sin descripción disponible.")

                oc = r.get("ocups") if isinstance(r.get("ocups"), dict) else {}
                has_occ = any(_is_num(v) for v in oc.values())

                occ_chips = kpi_html(r)
                if has_occ and occ_chips:
                    occ_block = f"""
                        <div class='kpis-head'>Índices de saturación disponibles</div>
                        <div class='kpis'>
                            {occ_chips}
                            <div class='kpi2' title='Porcentaje de similitud de perfil'>Similitud: <b>{si}</b></div>
                        </div>
                    """
                else:
                    occ_block = f"""
                        <div class='kpis-head'>Índices de saturación disponibles</div>
                        <div class='kpis'>
                            <div class='kpi'>No hay datos disponibles de ocupación</div>
                            <div class='kpi2' title='Porcentaje de similitud de perfil'>Similitud: <b>{si}</b></div>
                        </div>
                    """

                opiniones_block = reviews_html(r)

                return f"""
                    <div class='desc-card{css_sel}'>
                        <div class='header-row'>
                            <div class='title-block'>
                                <div class='card-title'>{zona}</div>
                                <div class='meta'>{ca} · {pr}</div>
                            </div>
                            <div class='badges'>
                                <span class='badge'>Zona turística</span>
                                {("<span class='badge'>Seleccionada</span>" if r.get("seleccionada") else "")}
                            </div>
                        </div>
                        {occ_block}
                        <p class='desc-body'>{desc}</p>
                        {opiniones_block}
                    </div>
                """

            cards_html = "\n".join([card_html(r) for r in ordered])
            full_html = css_block + f"<div>{cards_html}</div>"

            base_per_card = 220
            extra_if_reviews = 150
            cnt_reviews = sum(1 for r in ordered if r.get("opiniones"))
            est_height = base_per_card * len(ordered) + extra_if_reviews * cnt_reviews
            est_height = max(200, min(est_height+150, 2000))
            html_component(full_html, height=est_height, scrolling=True)

        if buscar:
            if df_fore is None:
                st.error("No hay forecasts disponibles; no se puede generar el ranking.")
            else:
                try:
                    indice_zona = zona_nombres.index(zona_objetivo)
                except ValueError:
                    st.error("No se encontró la zona seleccionada en los datos.")
                else:
                    n_total = len(df_knn)
                    n_vecinos = min(k_recom + 1, max(1, n_total))

                    zona_vector = df_knn.iloc[[indice_zona]]
                    Xq = knn_pipeline.named_steps['preprocessor'].transform(zona_vector)
                    distancias, indices = knn_pipeline.named_steps['knn'].kneighbors(Xq, n_neighbors=n_vecinos)

                    similares = []
                    for j, i in enumerate(indices[0]):
                        nombre = zona_nombres[i]
                        dist = float(distancias[0][j])
                        similares.append({"Zona": nombre, "Distancia": dist})

                    # dedup y p95
                    seen, filtrados = set(), []
                    for row in similares:
                        if row["Zona"] in seen:
                            continue
                        seen.add(row["Zona"])
                        filtrados.append(row)

                    dists = [r["Distancia"] for r in filtrados if r["Zona"] != zona_objetivo]
                    p95 = p95_normalized_similarity(dists)

                    # similitud
                    df_sim = pd.DataFrame([{
                        "Zona": r["Zona"],
                        "Similitud_num": 100.0 if r["Zona"] == zona_objetivo else 100.0 * (1.0 - (r["Distancia"] / p95))
                    } for r in filtrados])
                    df_sim["Similitud_num"] = df_sim["Similitud_num"].clip(0, 100)

                    # ocupación desglosada
                    zonas_list = df_sim["Zona"].astype(str).tolist()
                    occ_break = attach_occupancy_breakdown(df_fore, zonas_list, año_sel, mes_sel)

                    # ocupación media (para ordenar)
                    def occ_media(z):
                        vals = [v for v in occ_break.get(z, {}).values() if v is not None]
                        return float(np.mean(vals)) if vals else np.nan

                    df_sim["OCC_MEDIA"] = df_sim["Zona"].apply(occ_media)

                    # ordenar: mayor similitud, menor ocupación media
                    df_final = df_sim.sort_values(
                        by=["Similitud_num", "OCC_MEDIA"],
                        ascending=[False, True],
                        na_position="last"
                    ).reset_index(drop=True)

                    rows = []
                    for _, r in df_final.iterrows():
                        z = str(r["Zona"])
                        ca, pr = get_loc_info(z)
                        rows.append({
                            "zona": z,
                            "comunidad": ca,
                            "provincia": pr,
                            "ocups": occ_break.get(z, {}),
                            "similitud": f"{abs(r['Similitud_num']):.1f}%",
                            "desc": DESC_MAP.get(z, "Sin descripción disponible."),
                            "seleccionada": (z == zona_objetivo),
                            "opiniones": OPINIONES_MAP.get(z, [])
                        })

                    render_zone_result_cards(rows, subtitle=f"Ranking – {mes_nombre} {año_sel}")

elif opcion == "Ver mapas de saturación":
    st.subheader("Mapa de saturación turística por zona")
    st.info("Visualiza la concentración de turistas en cada zona. Filtra por zona y desplázate con el ratón para obtener una vista detallada.")

    # === Filtros temporales y tipo de turismo ===
    st.markdown("### 🎚️ Filtros temporales y tipo de turismo")
    col_f1, col_f2, col_f3 = st.columns([1.2, 1.2, 2])

    with col_f1:
        años_disponibles = sorted(df["AÑO"].dropna().unique())
        opciones_año = ["Todos los años"] + [str(a) for a in años_disponibles]
        año_seleccionado = st.selectbox("📅 Año", opciones_año)

    with col_f2:
        meses_disponibles = sorted(df["MES"].dropna().unique())
        opciones_mes = ["Todos los meses"] + [MESES_ES[m] for m in meses_disponibles if m in MESES_ES]
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

    df_filtrado = df.copy()
    if año_seleccionado != "Todos los años":
        df_filtrado = df_filtrado[df_filtrado["AÑO"] == int(año_seleccionado)]
    if mes_seleccionado != "Todos los meses":
        mes_num = [k for k, v in MESES_ES.items() if v == mes_seleccionado][0]
        df_filtrado = df_filtrado[df_filtrado["MES"] == mes_num]

    if columnas_seleccionadas:
        df_filtrado = _coerce_numeric(df_filtrado, columnas_seleccionadas)
        df_filtrado["viajeros"] = df_filtrado[columnas_seleccionadas].sum(axis=1).astype(float)
    else:
        df_filtrado["viajeros"] = 0.0

    df_grouped = df_filtrado.groupby(["ZONA_TURISTICA", "lat", "long", "AÑO", "MES"], as_index=False)["viajeros"].sum()
    df_grouped = df_grouped[df_grouped["viajeros"] > 0]
    df_grouped["viajeros_fmt"] = df_grouped["viajeros"].apply(lambda x: f"{x:,.0f}".replace(",", "."))
    df_grouped["anio_fmt"] = df_grouped["AÑO"].astype(str)
    df_grouped["mes_fmt"] = df_grouped["MES"].apply(lambda m: MESES_ES.get(m, str(m)))

    zonas = sorted(df_grouped["ZONA_TURISTICA"].unique()) if len(df_grouped) else []
    zona_sel = st.selectbox("Zona turística", ["Todas"] + zonas, index=0)

    view_state = pdk.ViewState(latitude=36, longitude=-3.5, zoom=3.9, pitch=40)

    color_defecto = hex_to_rgba(COLORS["indigo_dye"], alpha=0.75)
    color_seleccion = hex_to_rgba("#f59e0b", alpha=0.95)

    if zona_sel != "Todas" and len(df_grouped):
        fila = df_grouped[df_grouped["ZONA_TURISTICA"] == zona_sel].iloc[0]
        view_state = pdk.ViewState(
            latitude=float(fila["lat"]),
            longitude=float(fila["long"]),
            zoom=7.5,
            pitch=40
        )
        df_grouped = df_grouped.assign(
            fill_color=df_grouped["ZONA_TURISTICA"].apply(
                lambda z: color_seleccion if z == zona_sel else color_defecto
            )
        )
    else:
        df_grouped = df_grouped.assign(fill_color=[color_defecto] * len(df_grouped))

    max_v = float(df_grouped["viajeros"].max()) if len(df_grouped) else 1.0
    max_v = max(1.0, max_v)
    elevation_scale = 500000.0 / max_v

    line_rgba = hex_to_rgba("#000000", alpha=0.15)
    tooltip_bg = COLORS["anti_flash_white"]
    tooltip_text = COLORS["indigo_dye"]
    tooltip_border = "#a3bfd2"

    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=view_state,
        layers=[
            pdk.Layer(
                "ColumnLayer",
                data=df_grouped,
                get_position='[long, lat]',
                get_elevation='viajeros',
                elevation_scale=elevation_scale,
                radius=10000,
                extruded=True,
                get_fill_color='fill_color',
                get_line_color=line_rgba,
                pickable=True,
                auto_highlight=True,
            )
        ],
        tooltip={
            "html": f"""
                <div style="
                    font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
                    font-size: 12.5px;
                    line-height: 1.35;
                    min-width: 220px;
                ">
                    <div style="font-weight: 600; font-size: 14px; color:{tooltip_text}; margin-bottom: 6px;">
                        {{ZONA_TURISTICA}}
                    </div>
                    <div style="margin-bottom: 8px;">
                        <span style="
                            display: inline-block;
                            padding: 2px 8px;
                            border-radius: 999px;
                            background: rgba(0,0,0,0.04);
                            border: 1px solid {tooltip_border};
                            color: {tooltip_text};
                            font-size: 12px;
                            font-weight: 500;
                        ">
                            {{mes_fmt}} {{anio_fmt}}
                        </span>
                    </div>
                    <div style="
                        display: grid;
                        grid-template-columns: 1fr auto;
                        gap: 8px;
                        align-items: center;
                    ">
                        <div style="opacity: 0.75; color:{tooltip_text};">
                            Viajeros
                        </div>
                        <div style="
                            font-weight: 700;
                            color:{tooltip_text};
                            font-size: 14px;
                            letter-spacing: 0.2px;
                            font-variant-numeric: tabular-nums;
                        ">
                            {{viajeros_fmt}}
                        </div>
                    </div>
                </div>
            """,
            "style": {
                "backgroundColor": tooltip_bg,
                "color": tooltip_text,
                "border": f"1px solid {tooltip_border}",
                "borderRadius": "10px",
                "padding": "12px",
                "boxShadow": "0 4px 16px rgba(0,0,0,.08)"
            }
        }
    ))

elif opcion == "Encuentra tu destino":
    st.subheader("🧭 Encuentra tu destino")
    st.info("Filtra por características y descubre que destino se ajusta más a tus preferencias.")

    if _err_ops_global:
        st.warning(_err_ops_global)

    df_zt = df_zt_all.copy()
    # features coherentes con el modelo
    features = [
        "Tipo_Ubicación","Clima_Köppen","Estacionalidad_Climática","Nivel_Infraestructura_Turística",
        "Aeropuerto_mas_cercano","Tipo_Turismo_Principal","Actividad principal 1","Actividad principal 2",
        "Tipo_entorno_protegido","Patrimonio_cultural","Oferta_complementaria",
        "Altitud_Media_msnm","Distancia_al_mar_km","Indice_conectividad","Distancia_aeropuerto_km",
        "Distancia_estacion_tren_km","Porcentaje_area_protegida",
        "Actividad_Naturaleza","Actividad_Historico","Actividad_Entretenimiento","Actividad_Montanismo",
        "Actividad_Deportes_Acuaticos","Actividad_Gastronomia","Actividad_Cultural","Actividad_Ocio",
        "Actividad_Senderismo","Actividad_Turismo_rural","Actividad_Astronomia","Actividad_Deportes_de_Invierno",
        "Actividad_Observacion_de_Fauna","Actividad_Playa","Actividad_Cicloturismo","Actividad_Wellness_Termalismo",
        "Actividad_Compras","Actividad_Enoturismo","Actividad_Negocios_MICE","Actividad_Religioso","Actividad_Aventura",
        "Actividad_Turismo_Nautico",
    ]
    for col in features:
        if col in df_zt.columns and df_zt[col].dtype == object:
            df_zt[col] = df_zt[col].astype(str)
    nombre_col = 'ZONA_TURISTICA' if 'ZONA_TURISTICA' in df_zt.columns else df_zt.columns[0]

    def safe_options(df, col):
        if col in df.columns:
            return sorted(pd.Series(df[col]).dropna().astype(str).unique().tolist())
        return []

    def altitud_range(df, col):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if not s.empty:
                lo, hi = int(s.min()), int(s.max())
                span = max(hi - lo, 1)
                step = max(1, round(span / 200))
                return lo, hi, step
        return None

    opts_tipo_ubic = safe_options(df_zt, "Tipo_Ubicación")
    opts_clima     = safe_options(df_zt, "Clima_Köppen")
    opts_tipo_tur  = safe_options(df_zt, "Tipo_Turismo_Principal")
    opts_estac     = safe_options(df_zt, "Estacionalidad_Climática")
    opts_infra     = safe_options(df_zt, "Nivel_Infraestructura_Turística")
    opts_act1      = safe_options(df_zt, "Actividad principal 1")
    opts_act2      = safe_options(df_zt, "Actividad principal 2")
    alt_info       = altitud_range(df_zt, "Altitud_Media_msnm")

    defaults = {
        "k_tipo_ubic": [],
        "k_clima":     [],
        "k_tipo_tur":  [],
        "k_estac":     [],
        "k_infra":     [],
        "k_act1":      [],
        "k_act2":      [],
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    if alt_info:
        alt_min, alt_max, step = alt_info
        st.session_state.setdefault("k_alt_sel", (alt_min, alt_max))

    colA, colB, colC = st.columns([1, 1, 1], gap="large")

    with colA:
        tipo_ubic = st.multiselect(
            "Tipo de ubicación",
            opts_tipo_ubic,
            placeholder="Elige una opción",
            key="k_tipo_ubic",
            help="Selecciona el contexto geográfico principal del destino."
        )
        clima = st.multiselect(
            "Clima (Köppen)",
            opts_clima,
            placeholder="Elige una opción",
            key="k_clima",
            help="Clasificación climática de Köppen."
        )
        tipo_tur = st.multiselect(
            "Tipo de turismo principal",
            opts_tipo_tur,
            placeholder="Elige una opción",
            key="k_tipo_tur",
            help="Interés dominante del viaje."
        )

    with colB:
        act1 = st.multiselect(
            "Actividad principal",
            opts_act1,
            placeholder="Elige una opción",
            key="k_act1",
            help="Actividad estrella del destino."
        )
        act2 = st.multiselect(
            "Actividad secundaria",
            opts_act2,
            placeholder="Elige una opción",
            key="k_act2",
            help="Actividades complementarias disponibles."
        )
        infra = st.multiselect(
            "Nivel de infraestructura turística",
            opts_infra,
            placeholder="Elige una opción",
            key="k_infra",
            help="Grado de desarrollo turístico (bajo, medio, alto)."
        )

    with colC:
        estac = st.multiselect(
            "Estacionalidad climática",
            opts_estac,
            placeholder="Elige una opción",
            key="k_estac",
            help="Patrón de estaciones relevante para la experiencia."
        )

        if alt_info:
            alt_min, alt_max, step = alt_info
            alt_sel = st.slider(
                "Altitud media (msnm)",
                min_value=alt_min,
                max_value=alt_max,
                step=step,
                key="k_alt_sel",
                help="Rango de altitud promedio sobre el nivel del mar."
            )
        else:
            alt_sel = None
            st.caption("No hay datos de altitud disponibles. "
                    "Tip: Si añades altitudes a la fuente de datos, aquí podrás filtrar por msnm.")

    chips = []
    def chip(name, vals):
        if vals:
            chips.append(f"**{name}:** {', '.join(map(str, vals))}")
    chip("Tipo de ubicación", st.session_state["k_tipo_ubic"])
    chip("Clima",             st.session_state["k_clima"])
    chip("Turismo",           st.session_state["k_tipo_tur"])
    chip("Estacionalidad",    st.session_state["k_estac"])
    chip("Infraestructura",   st.session_state["k_infra"])
    chip("Actividad 1",       st.session_state["k_act1"])
    chip("Actividad 2",       st.session_state["k_act2"])
    if alt_info and st.session_state["k_alt_sel"] != (alt_min, alt_max):
        lo, hi = st.session_state["k_alt_sel"]
        chips.append(f"**Altitud:** {lo}–{hi} msnm")

    st.markdown("##### Selección actual" if chips else "")
    st.markdown(" · ".join(chips) if chips else "Sin filtros aplicados por ahora.")

    df_fore, err_fore = df_fore_global, _err_fore_global
    if err_fore:
        st.error(err_fore)

    colO1, colO2, colO3 = st.columns([2, 2, 1])
    with colO1:
        año_sel = st.selectbox("Año", [2025, 2026, 2027], index=0)
    with colO2:
        mes_nombre = st.selectbox("Mes", list(MESES_ES.values()), index=0)
        mes_sel = [k for k, v in MESES_ES.items() if v == mes_nombre][0]
    with colO3:
        k_sugerencias = st.slider("N.º de sugerencias (si no hay coincidencias)", 3, 10, 6, 1)

    fallback_similares = True

    if st.button("🔎 Buscar destinos", use_container_width=True):
        df_fil = df_zt.copy()

        def filtrar_in(df_fil, col, valores):
            if valores and col in df_fil.columns:
                return df_fil[df_fil[col].isin(valores)]
            return df_fil

        df_fil = filtrar_in(df_fil, "Tipo_Ubicación", tipo_ubic)
        df_fil = filtrar_in(df_fil, "Clima_Köppen", clima)
        df_fil = filtrar_in(df_fil, "Tipo_Turismo_Principal", tipo_tur)
        df_fil = filtrar_in(df_fil, "Estacionalidad_Climática", estac)
        df_fil = filtrar_in(df_fil, "Nivel_Infraestructura_Turística", infra)
        df_fil = filtrar_in(df_fil, "Actividad principal 1", act1)
        df_fil = filtrar_in(df_fil, "Actividad principal 2", act2)

        if alt_sel and 'Altitud_Media_msnm' in df_fil.columns:
            df_fil = df_fil[pd.to_numeric(df_fil['Altitud_Media_msnm'], errors='coerce').between(alt_sel[0], alt_sel[1])]

        def build_query_from_filters():
            q = {}
            def pick_cat(col, seleccion):
                if seleccion:
                    return str(seleccion[0])
                if col in df_zt.columns and df_zt[col].notna().any():
                    return str(df_zt[col].mode(dropna=True).iloc[0])
                return ""
            def pick_num(col, default=0.0):
                if col in df_zt.columns:
                    series = pd.to_numeric(df_zt[col], errors="coerce")
                    if series.notna().any():
                        return float(series.median())
                return float(default)

            q['Tipo_Ubicación'] = pick_cat('Tipo_Ubicación', tipo_ubic)
            q['Clima_Köppen'] = pick_cat('Clima_Köppen', clima)
            q['Tipo_Turismo_Principal'] = pick_cat('Tipo_Turismo_Principal', tipo_tur)
            q['Estacionalidad_Climática'] = pick_cat('Estacionalidad_Climática', estac)
            q['Nivel_Infraestructura_Turística'] = pick_cat('Nivel_Infraestructura_Turística', infra)
            q['Actividad principal 1'] = pick_cat('Actividad principal 1', act1)
            q['Actividad principal 2'] = pick_cat('Actividad principal 2', act2)
            q['Aeropuerto_mas_cercano'] = pick_cat('Aeropuerto_mas_cercano', [])
            q['Tipo_entorno_protegido'] = pick_cat('Tipo_entorno_protegido', [])
            q['Patrimonio_cultural'] = pick_cat('Patrimonio_cultural', [])
            q['Oferta_complementaria'] = pick_cat('Oferta_complementaria', [])

            num_cols = ["Altitud_Media_msnm","Distancia_al_mar_km","Indice_conectividad",
                        "Distancia_aeropuerto_km","Distancia_estacion_tren_km","Porcentaje_area_protegida"]
            q['Altitud_Media_msnm'] = float(np.mean(alt_sel)) if alt_sel and 'Altitud_Media_msnm' in df_zt.columns else pick_num('Altitud_Media_msnm')
            for nc in num_cols[1:]:
                q[nc] = pick_num(nc)

            for ac in [c for c in features if c.startswith("Actividad_")]:
                q[ac] = 0
            return pd.DataFrame([q])

        knn_pipeline, df_knn, _ = entrenar_pipeline(df_zt)
        q_df = build_query_from_filters()

        def render_zone_result_cards(rows: list[dict], subtitle: str = "", per_row: int = 1):
            if not rows:
                return

            colA, colB, colC = st.columns([9, 13, 3])
            with colA:
                st.markdown(f"#### {subtitle}")

            with colC:
                logo_path = LOGOS_DIR / "TripAdvisor_Logo.svg"
                if logo_path.exists():
                    st.markdown(
                        "<div style='text-align:left;font-size:0.85rem;color:#6b7280;'>Powered by</div>",
                        unsafe_allow_html=True
                    )
                    st.image(str(logo_path), width=120)

            css_block = """
            <style>
            .desc-card{
              background:#fff;border:1px solid #d9e2ea;border-radius:16px;
              padding:14px 16px;box-shadow:0 4px 14px rgba(0,0,0,0.06);
              font-family:Satoshi,system-ui;overflow:visible;margin-bottom:10px;
            }
            .sel{border:2px solid #306388;}
            .header-row{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:6px;}
            .title-block{display:flex;flex-direction:column;min-width:0;}
            .badges{display:flex;gap:8px;flex-wrap:wrap;margin:0;}
            .badge{background:#e6eaed;color:#224762;border-radius:999px;padding:2px 10px;font-size:.75rem;font-weight:600;}
            .card-title{font-weight:800;color:#224762;margin:0 0 2px 0;font-size:1.05rem;}
            .meta{color:#4a5a67;font-size:.92rem;margin:0;}
            .kpis-head{font-weight:700;color:#224762;font-size:.88rem;margin:4px 0 6px 0;}
            .kpis{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 8px 0;}
            .kpi{background:#f5f7f9;border:1px solid #e5eef5;border-radius:10px;padding:6px 10px;font-size:.86rem;color:#224762;}
            .kpi2{background:#e9eff3;border:1px solid #e5eef5;border-radius:3px;padding:6px 10px;font-size:.86rem;color:#224762;}
            .desc-body{color:#3a4b59;margin:0;line-height:1.35;font-size:0.95rem;}
            .reviews-wrap { margin-top:8px; }
            .reviews-wrap details { background:#f7f9fb; border:1px solid #e5eef5; border-radius:10px; padding:8px 10px; }
            .reviews-wrap summary { cursor:pointer; color:#224762; font-weight:700; }
            .reviews-scroll{ max-height:500px; overflow:auto; margin-top:8px; padding-right:6px; }
            .reviews-list { margin:0 0 0 16px; color:#3a4b59; font-size:0.92rem; }
            .reviews-list li { margin-bottom:6px; }
            </style>
            """

            sel_rows = [r for r in rows if r.get("seleccionada")]
            other_rows = [r for r in rows if not r.get("seleccionada")]
            ordered = sel_rows + other_rows

            def _is_num(x):
                try:
                    return (x is not None) and (not (isinstance(x, float) and np.isnan(x)))
                except Exception:
                    return False

            def kpi_html(r: dict) -> str:
                oc = r.get("ocups")
                if not isinstance(oc, dict) or not len(oc):
                    return ""
                parts = []
                for tipo in [t for t in ["Hotel", "Turismo rural", "Apartamentos", "Camping"] if t in oc]:
                    if _is_num(oc.get(tipo)):
                        parts.append(f"<div class='kpi'>{OCC_LABELS[tipo]}: <b>{format_pct(oc[tipo])}</b></div>")
                return "".join(parts)

            def reviews_html(r: dict) -> str:
                opiniones = r.get("opiniones") or []
                if not opiniones:
                    return ""
                try:
                    n = min(5, len(opiniones))
                    idx = np.random.choice(len(opiniones), size=n, replace=False)
                    sample = [str(opiniones[i]) for i in idx]
                except Exception:
                    sample = [str(op) for op in opiniones[:5]]

                items = "\n".join([f"<li>{html.escape(s)}</li>" for s in sample])
                return f"""
                    <div class='reviews-wrap'>
                        <details>
                            <summary>🌟 Opiniones de viajeros</summary>
                            <div class='reviews-scroll'>
                                <ul class='reviews-list'>
                                    {items}
                                </ul>
                            </div>
                        </details>
                    </div>
                """

            def card_html(r: dict) -> str:
                css_sel = " sel" if r.get("seleccionada") else ""
                zona = r.get("zona", "—")
                ca = r.get("comunidad", "—")
                pr = r.get("provincia", "—")
                si = r.get("similitud") or "No hay datos"
                desc = r.get("desc", "Sin descripción disponible.")

                oc = r.get("ocups") if isinstance(r.get("ocups"), dict) else {}
                has_occ = any(_is_num(v) for v in oc.values())

                occ_chips = kpi_html(r)
                if has_occ and occ_chips:
                    occ_block = f"""
                        <div class='kpis-head'>Índices de saturación disponibles</div>
                        <div class='kpis'>
                            {occ_chips}
                            <div class='kpi2' title='Porcentaje de similitud de perfil'>Similitud: <b>{si}</b></div>
                        </div>
                    """
                else:
                    occ_block = f"""
                        <div class='kpis-head'>Índices de saturación disponibles</div>
                        <div class='kpis'>
                            <div class='kpi'>No hay datos disponibles de ocupación</div>
                            <div class='kpi2' title='Porcentaje de similitud de perfil'>Similitud: <b>{si}</b></div>
                        </div>
                    """

                opiniones_block = reviews_html(r)

                return f"""
                    <div class='desc-card{css_sel}'>
                        <div class='header-row'>
                            <div class='title-block'>
                                <div class='card-title'>{zona}</div>
                                <div class='meta'>{ca} · {pr}</div>
                            </div>
                            <div class='badges'>
                                <span class='badge'>Zona turística</span>
                                {("<span class='badge'>Seleccionada</span>" if r.get("seleccionada") else "")}
                            </div>
                        </div>
                        {occ_block}
                        <p class='desc-body'>{desc}</p>
                        {opiniones_block}
                    </div>
                """

            cards_html = "\n".join([card_html(r) for r in rows])
            full_html = css_block + f"<div>{cards_html}</div>"

            base_per_card = 220
            extra_if_reviews = 150
            cnt_reviews = sum(1 for r in rows if r.get("opiniones"))
            est_height = base_per_card * len(rows) + extra_if_reviews * cnt_reviews
            est_height = max(200, min(est_height+150, 2000))
            html_component(full_html, height=est_height, scrolling=True)

        if len(df_fil) > 0:
            st.success(f"Se han encontrado {len(df_fil)} destinos que cumplen tus criterios.")

            dist_all, idx_all = entrenar_pipeline(df_zt)[0].named_steps['knn'].kneighbors(
                entrenar_pipeline(df_zt)[0].named_steps['preprocessor'].transform(q_df),
                n_neighbors=len(entrenar_pipeline(df_zt)[1])
            )
            nombres_all = df_zt[nombre_col].astype(str).tolist()
            dist_map = {nombres_all[i]: float(dist_all[0][j]) for j, i in enumerate(idx_all[0])}

            zonas_list = df_fil[nombre_col].astype(str).tolist()

            dists_found = [dist_map.get(str(z), np.nan) for z in zonas_list]
            dists_found = [d for d in dists_found if pd.notna(d)]
            p95 = p95_normalized_similarity(dists_found)

            occ_break = attach_occupancy_breakdown(df_fore, zonas_list, año_sel, mes_sel)

            rows = []
            for z in zonas_list:
                ca, pr = get_loc_info(z)
                dist = dist_map.get(z, None)
                sim = (100.0 * (1.0 - dist / p95)) if (dist is not None) else None

                vals = [v for v in occ_break.get(z, {}).values() if v is not None]
                occ_med = float(np.mean(vals)) if vals else np.nan

                rows.append({
                    "zona": z,
                    "comunidad": ca,
                    "provincia": pr,
                    "ocups": occ_break.get(z, {}),
                    "similitud": f"{abs(sim):.1f}%" if sim is not None else "—",
                    "desc": DESC_MAP.get(z, "Sin descripción disponible."),
                    "seleccionada": False,
                    "_occ_media": occ_med,
                    "opiniones": OPINIONES_MAP.get(z, [])
                })

            rows_sorted = sorted(
                rows,
                key=lambda r: (-(float(r["similitud"][:-1]) if r["similitud"] != "—" else -0.0),
                               (r["_occ_media"] if not np.isnan(r["_occ_media"]) else 9999.0))
            )
            for r in rows_sorted:
                r.pop("_occ_media", None)

            render_zone_result_cards(rows_sorted, subtitle=f"Resultados – {mes_nombre} {año_sel}")

        else:
            if not fallback_similares:
                st.warning("No se han encontrado destinos con esos criterios. Activa la casilla de sugerencias para ver alternativas similares.")
            else:
                st.info("No hubo coincidencias exactas. Mostrando destinos similares a tus preferencias.")
                knn_pipeline, df_knn, _ = entrenar_pipeline(df_zt)
                n_total = len(df_knn)
                n_vecinos = min(k_sugerencias, max(1, n_total))

                Xq = knn_pipeline.named_steps['preprocessor'].transform(q_df)
                dist, idx = knn_pipeline.named_steps['knn'].kneighbors(Xq, n_neighbors=n_vecinos)

                nombres = df_zt[nombre_col].astype(str).tolist()
                zonas_list = [str(nombres[i]) for j, i in enumerate(idx[0])]
                occ_break = attach_occupancy_breakdown(df_fore, zonas_list, año_sel, mes_sel)

                dists = [float(dist[0][j]) for j, _ in enumerate(idx[0])]
                p95 = p95_normalized_similarity(dists)

                rows = []
                for j, i in enumerate(idx[0]):
                    z = str(nombres[i])
                    ca, pr = get_loc_info(z)
                    sim = (100 * (1 - float(dist[0][j]) / p95))
                    vals = [v for v in occ_break.get(z, {}).values() if v is not None]
                    occ_med = float(np.mean(vals)) if vals else np.nan
                    rows.append({
                        "zona": z,
                        "comunidad": ca,
                        "provincia": pr,
                        "ocups": occ_break.get(z, {}),
                        "similitud": f"{abs(sim):.1f}%",
                        "desc": DESC_MAP.get(z, "Sin descripción disponible."),
                        "seleccionada": False,
                        "_occ_media": occ_med,
                        "opiniones": OPINIONES_MAP.get(z, [])
                    })

                rows = sorted(
                    rows,
                    key=lambda r: (-(float(r["similitud"][:-1]) if r["similitud"] != "—" else 0.0),
                                   (r["_occ_media"] if not np.isnan(r["_occ_media"]) else 9999.0))
                )
                rows = rows[:k_sugerencias]
                for r in rows:
                    r.pop("_occ_media", None)

                render_zone_result_cards(rows, subtitle=f"Sugerencias – {mes_nombre} {año_sel}")

elif opcion == "Consultar datos históricos":
    st.subheader("📈 Datos históricos del turismo")
    st.caption("Analiza la evolución temporal por zona turística, tipo de alojamiento y periodo.")

    columnas_tipo = {
        "Turismo Hotelero": "VIAJEROS_EOH",
        "Turismo Rural": "VIAJEROS_EOTR",
        "Apartamentos": "VIAJEROS_EOAP",
        "Campings": "VIAJEROS_EOAC"
    }

    c1, c2, c3 = st.columns([1.5, 2.5, 2.5])
    with c1:
        años = sorted(df["AÑO"].dropna().unique())
        if not len(años):
            st.warning("No hay datos de años en el dataset.")
            st.stop()
        año_min, año_max = int(min(años)), int(max(años))
        año_opciones = ["Todos"] + [str(a) for a in años]
        año_sel = st.selectbox("Años", options=año_opciones, index=0)
        año_rango = (año_min, año_max) if año_sel == "Todos" else (int(año_sel), int(año_sel))
    with c2:
        meses_disponibles = sorted(df["MES"].dropna().unique().tolist())
        meses_labels = [MESES_ES.get(m, str(m)) for m in meses_disponibles]
        meses_opciones = ["Todos"] + meses_labels
        mes_sel = st.selectbox("Meses", options=meses_opciones, index=0)
        meses_sel = meses_disponibles if mes_sel == "Todos" else [k for k,v in MESES_ES.items() if v == mes_sel]
    with c3:
        tipos_all = list(columnas_tipo.keys())
        tipos_opciones = ["Todos"] + tipos_all
        tipo_sel = st.selectbox("Tipo de alojamiento", options=tipos_opciones, index=0)
        tipos_sel = tipos_all if tipo_sel == "Todos" else [tipo_sel]

    c4, c5 = st.columns([2.5, 3])
    with c4:
        zonas_all = sorted(df["ZONA_TURISTICA"].dropna().unique().tolist())
        zonas_opciones = ["Todas"] + zonas_all
        zona_sel = st.selectbox("Zona turística", options=zonas_opciones, index=0)
        zonas_sel = zonas_all if zona_sel == "Todas" else [zona_sel]
    with c5:
        nivel = st.radio("Agregación temporal", ["Mensual", "Trimestral", "Anual"], horizontal=True)

    if not tipos_sel:
        st.warning("Selecciona al menos un tipo de alojamiento.")
        st.stop()

    cols_metric = [columnas_tipo[t] for t in tipos_sel]
    df_h = df.copy()
    df_h = df_h[(df_h["AÑO"].between(año_rango[0], año_rango[1])) & (df_h["MES"].isin(meses_sel))]
    if zonas_sel:
        df_h = df_h[df_h["ZONA_TURISTICA"].isin(zonas_sel)]

    df_h = _coerce_numeric(df_h, cols_metric)
    df_h["VIAJEROS_SEL"] = df_h[cols_metric].sum(axis=1)

    df_h["FECHA"] = pd.to_datetime(df_h["AÑO"].astype(int).astype(str) + "-" + df_h["MES"].astype(int).astype(str) + "-01")
    df_h["TRIM"] = pd.PeriodIndex(df_h["FECHA"], freq="Q").astype(str)

    group_keys = ["ZONA_TURISTICA"]
    if nivel == "Mensual":
        group_keys += ["AÑO", "MES", "FECHA"]
        x_field, x_title = "FECHA", "Fecha"
    elif nivel == "Trimestral":
        group_keys += ["AÑO", "TRIM"]
        x_field, x_title = "TRIM", "Trimestre"
    else:
        group_keys += ["AÑO"]
        x_field, x_title = "AÑO", "Año"

    agg = df_h.groupby(group_keys, as_index=False)["VIAJEROS_SEL"].sum()

    total_periodo = int(agg["VIAJEROS_SEL"].sum()) if len(agg) else 0

    hoy = datetime.now(ZoneInfo("Europe/Madrid"))
    mes_actual = hoy.month
    yoy_txt = "N/D"
    try:
        base = df.copy()
        if zonas_sel:
            base = base[base["ZONA_TURISTICA"].isin(zonas_sel)]
        if isinstance(año_rango, tuple):
            base = base[base["AÑO"].between(año_rango[0], año_rango[1])]
        base = _coerce_numeric(base, cols_metric)
        base["VIAJEROS_SEL"] = base[cols_metric].sum(axis=1)
        años_disp = sorted(base["AÑO"].dropna().unique())
        if len(años_disp) >= 2:
            ult_anio = int(años_disp[-1])
            ant_anio = int(años_disp[-2])
            max_mes_ult = int(base.loc[base["AÑO"] == ult_anio, "MES"].max())
            corte_mes = min(max(mes_actual - 1, 1), max_mes_ult)
            ytd_ult = int(base.loc[(base["AÑO"] == ult_anio) & (base["MES"].between(1, corte_mes)), "VIAJEROS_SEL"].sum())
            ytd_ant = int(base.loc[(base["AÑO"] == ant_anio) & (base["MES"].between(1, corte_mes)), "VIAJEROS_SEL"].sum())
            if ytd_ant > 0:
                yoy = (ytd_ult / ytd_ant - 1) * 100
                yoy_txt = f"{yoy:+.1f}%"
    except Exception:
        yoy_txt = "N/D"

    top_zona_txt = "N/D"
    if len(df_h):
        top_zona = df_h.groupby("ZONA_TURISTICA", as_index=False)["VIAJEROS_SEL"].sum().sort_values("VIAJEROS_SEL", ascending=False).head(1)
        if len(top_zona):
            top_zona_txt = f"{top_zona.iloc[0]['ZONA_TURISTICA']} ({int(top_zona.iloc[0]['VIAJEROS_SEL']):,}".replace(",", ".") + ")"

    k1, k2, k3 = st.columns([3, 3, 6])
    k1.metric("Viajeros", f"{total_periodo:,}".replace(",", "."))
    k2.metric("Variación YoY (último año vs. anterior)", yoy_txt)
    k3.metric("Zona top", top_zona_txt)

    st.divider()

    st.markdown("#### Evolución temporal")
    agg_total = agg.groupby(x_field, as_index=False)["VIAJEROS_SEL"].sum().sort_values(x_field)
    base_total = alt.Chart(agg_total).mark_line(point=True).encode(
        x=alt.X(x_field, title=x_title, sort=None),
        y=alt.Y("VIAJEROS_SEL:Q", title="Viajeros"),
        tooltip=[x_field, alt.Tooltip("VIAJEROS_SEL:Q", title="Viajeros", format=",.0f")]
    ).properties(height=280)
    st.altair_chart(base_total.interactive(), use_container_width=True)

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
            st.altair_chart(line.interactive(), use_container_width=True)
        else:
            st.warning("No hay datos para las zonas seleccionadas.")

    st.divider()

    st.markdown("#### Calendario (Año × Mes)")
    df_hm = df_h.groupby(["AÑO", "MES"], as_index=False)["VIAJEROS_SEL"].sum()
    if len(df_hm):
        df_hm["Mes"] = df_hm["MES"].map(MESES_ES)
        heat = alt.Chart(df_hm).mark_rect().encode(
            x=alt.X("Mes:N", sort=list(MESES_ES.values()), title="Mes"),
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

    st.markdown("#### Top 10 zonas")
    topN = df_h.groupby("ZONA_TURISTICA", as_index=False)["VIAJEROS_SEL"].sum().sort_values("VIAJEROS_SEL", ascending=False).head(10)
    if len(topN):
        barchart = alt.Chart(topN).mark_bar().encode(
            x=alt.X("VIAJEROS_SEL:Q", title="Viajeros"),
            y=alt.Y("ZONA_TURISTICA:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("VIAJEROS_SEL:Q", title="Viajeros", format=",.0f"), "ZONA_TURISTICA:N"]
        ).properties(height=28*len(topN) + 20)
        st.altair_chart(barchart, use_container_width=True)

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
    st.subheader("Acerca del proyecto")
    
    def _img_data_uri(path: Path) -> str | None:
        try:
            data = Path(path).read_bytes()
            ext = str(path).lower().split(".")[-1]
            mime = (
                "image/png" if ext in ("png",)
                else "image/jpeg" if ext in ("jpg", "jpeg")
                else "image/svg+xml" if ext == "svg"
                else "image/png"
            )
            return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        except Exception:
            return None

    # Usa el LOGOS_DIR global ya definido en tu app
    equipo = [
        {
            "nombre": "Santiago Barbés Cañal",
            "estudios": "Estadística · UVa",
            "rol": "Data Engineer · SDG Group",
            "foto": LOGOS_DIR / "Foto_SantiagoBarbés.jpeg",
            "linkedin": "https://www.linkedin.com/in/santiago-barbes-canal/"
        },
        {
            "nombre": "Miguel Barrios Díaz",
            "estudios": "Matemáticas · UCM",
            "rol": "Analista de datos · Serveo",
            "foto": LOGOS_DIR / "Foto_Miguel.jpeg",
            "linkedin": "https://www.linkedin.com/in/miguel-barrios-diaz-bba4282a2/"
        },
        {
            "nombre": "Jorge González Perea",
            "estudios": "Física · UCM",
            "rol": "Analista de datos · SDG Group",
            "foto": LOGOS_DIR / "Foto_JorgeGonzalez.jpeg",
            "linkedin": "https://www.linkedin.com/in/jorge-gonzalez-perea/"
        },
        {
            "nombre": "Jorge Lavín",
            "estudios": "Estadística y Empresa · UC3M",
            "rol": "Analista de datos · MBD Analytics",
            "foto": LOGOS_DIR / "Foto_JorgeLavin.jpeg",
            "linkedin": "https://www.linkedin.com/in/jorge-lavin-212416234/"
        },
        {
            "nombre": "Francisco Javier Palau Cuenca",
            "estudios": "Matemáticas · UA",
            "rol": "Analista de datos · Management Solutions",
            "foto": LOGOS_DIR / "Foto_FranciscoPalau.jpeg",
            "linkedin": "https://www.linkedin.com/in/francisco-javier-palau-cuenca/"
        }
    ]
    col1, col2, col3 = st.columns([20, 10, 20])
    with col2:
        deep5 = LOGOS_DIR / "Logo_Deep5.svg"
        if deep5.exists():
            st.image(str(deep5), width=300)

    # Construir tarjetas
    cards_html = []
    for m in equipo:
        name = str(m.get("nombre","")).strip()
        estudios = html.escape(str(m.get("estudios","")).strip())
        rol = html.escape(str(m.get("rol","")).strip())
        foto_path = m.get("foto")
        linkedin = str(m.get("linkedin", "")).strip()

        data_uri = _img_data_uri(foto_path) if foto_path else None
        ini = "".join([p[0] for p in name.split()[:2]]).upper() if name else "–"
        name_safe = html.escape(name)

        avatar_html = (
            f'<img class="avatar" src="{data_uri}" alt="{name_safe}">'
            if data_uri else
            f'<div class="avatar avatar-fallback">{html.escape(ini)}</div>'
        )

        linkedin_logo = _img_data_uri(LOGOS_DIR / "LinkedIn_Logo.svg") 
        linkedin_html = ""
        if linkedin and linkedin_logo:
            linkedin_html = f"""
                <a class="team-link" href="{html.escape(linkedin)}" target="_blank" rel="noopener noreferrer">
                    <img src="{linkedin_logo}" class="linkedin-icon" alt="LinkedIn">
                </a>
            """

        card = f"""
            <div class="team-card">
                <div class="avatar-wrap">{avatar_html}</div>
                <div class="team-name">{name_safe}</div>
                <div class="team-role">{rol}</div>
                <div class="team-study">{estudios}</div>
                <div class="team-actions">{linkedin_html}</div>
            </div>
        """
        cards_html.append(card)

    # Estilos + render
    st.html(f"""
    <style>
    .team-wrap {{
        display:grid; grid-template-columns: repeat(5, 1fr);
        gap:16px; margin-top:10px;
    }}
    @media (max-width:1200px){{ .team-wrap{{grid-template-columns:repeat(3,1fr);}} }}
    @media (max-width:800px){{ .team-wrap{{grid-template-columns:repeat(2,1fr);}} }}
    @media (max-width:520px){{ .team-wrap{{grid-template-columns:1fr;}} }}

    .team-card {{
        background:#fff; border:1px solid #e5eef5; border-radius:14px;
        padding:16px 14px 18px; box-shadow:0 2px 10px rgba(0,0,0,0.05);
        text-align:center;
    }}
    .avatar-wrap {{ display:flex; justify-content:center; margin-bottom:12px; }}

    .avatar {{
        width:128px; height:128px; border-radius:50%;
        object-fit:cover; border:2px solid #e6eaed;
        box-shadow:0 2px 10px rgba(0,0,0,0.06);
    }}
    .avatar-fallback {{
        width:128px; height:128px; border-radius:50%;
        display:flex; align-items:center; justify-content:center;
        background: linear-gradient(135deg, #224762 0%, #306388 100%);
        color:#ffffff; font-weight:800; font-size:34px;
        box-shadow:0 2px 10px rgba(0,0,0,0.06);
    }}

    .team-name {{ font-weight:800; color:#224762; font-size:1.1rem; margin-top:2px; }}
    .team-role {{ color:#31414f; font-size:.96rem; margin-top:2px; }}
    .team-study {{ color:#4f6475; font-size:.9rem; margin-top:4px; }}

    .team-actions {{ margin-top:10px; display:flex; justify-content:center; }}
    .team-link {{
        display:inline-flex; align-items:center; justify-content:center;
        width:100px; height:35px;
        border-radius:8px;
        border:1px solid #d9e2ea;
        background:#fff;
        transition: all .15s ease;
    }}
    .team-link:hover {{
        background:#f5f7f9; border-color:#cfd9e2;
    }}
    .linkedin-icon {{
        width:100px; height:18px; display:block;
    }}

    /* Bloque de descripción ampliada */
    .about-card {{
        background:#fff; border:1px solid #e5eef5; border-radius:14px;
        padding:18px 20px; box-shadow:0 2px 10px rgba(0,0,0,0.05);
        margin-top:20px;
    }}
    .about-title {{
        margin:0 0 8px 0; font-weight:800; font-size:1.1rem; color:#224762;
    }}
    .about-text {{
        color:#31414f; font-size:.96rem; line-height:1.6; margin:0;
        text-align: justify;
    }}
    </style>

    <div class="team-wrap">
      {''.join(cards_html)}
    </div>

    <div class="about-card">
    <h4 class="about-title">Sobre el proyecto</h4>
    <p class="about-text">
        Este Trabajo de Fin de Máster traslada a un caso real todas las capacidades adquiridas durante la formación,
        aplicando un enfoque integral que abarca desde la construcción y depuración de fuentes de datos hasta la 
        generación de modelos de recomendación y visualización interactiva. A lo largo del desarrollo se han utilizado 
        herramientas de preparación como <b>Power Query</b> y <b>Pandas</b> para transformar y limpiar los datos, 
        garantizando consistencia y calidad en la información utilizada.
    </p>
    <p class="about-text" style="margin-top:10px;">
        En la fase analítica, se emplearon metodologías estadísticas y técnicas de <b>machine learning</b>, 
        entre ellas <b>modelos de similitud (k-NN)</b> para identificar destinos turísticos alternativos 
        y enfoques exploratorios para detectar patrones de comportamiento en la demanda. Este análisis se complementó 
        con indicadores clave que permiten interpretar de manera clara la evolución histórica y los riesgos de saturación 
        de determinadas zonas.
    </p>
    <p class="about-text" style="margin-top:10px;">
        La capa de interacción se desarrolló mediante <b>Streamlit</b>, lo que permitió implementar una plataforma 
        intuitiva y accesible, capaz de combinar gráficos dinámicos, mapas interactivos y un motor de recomendación 
        integrado. Gracias a ello, los usuarios pueden explorar la información de manera ágil, visualizando tanto 
        datos históricos como escenarios alternativos de redistribución turística.
    </p>
    <p class="about-text" style="margin-top:10px;">
        El proyecto también subraya la <b>importancia de la recolección sistemática de datos por parte de entidades públicas,
        organismos oficiales y organizaciones privadas</b>. Sin su labor de registro, actualización y estandarización, 
        sería imposible disponer de información fiable y comparable en el tiempo. Este esfuerzo colectivo pone en valor 
        la utilidad de los datos como activo estratégico, facilitando diagnósticos más precisos y la definición de políticas
        turísticas más efectivas.
    </p>
    <p class="about-text" style="margin-top:10px;">
        El objetivo final del proyecto es aportar una base clara y práctica para equilibrar la demanda turística, 
        ofreciendo <b>alternativas viables y medibles</b> que compatibilicen la experiencia del visitante, el 
        bienestar de la población residente y la conservación del entorno. La plataforma integra criterios de 
        <b>buen uso de los datos</b>, transparencia y lectura responsable de resultados, con un enfoque de 
        sostenibilidad y visión de largo plazo. 
    </p>
    <p class="about-text" style="margin-top:10px;">
        En conjunto, el proyecto demuestra cómo la integración de <b>herramientas de ciencia de datos, ingeniería, 
        visualización y gestión</b> puede traducirse en soluciones aplicables a un reto real de la sociedad, 
        alineando la innovación tecnológica con la sostenibilidad turística.
    </p>
    </div>


    """)


# =========================
# FOOTER
# =========================
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns([6, 2, 2, 2, 6])
with col2:
    deep5 = LOGOS_DIR / "Logo_Deep5.svg"
    if deep5.exists():
        st.image(str(deep5), width=160)
with col3:
    st.markdown("<p style='text-align:center; font-size:0.9rem; margin-top:20px;'>En colaboración con</p>", unsafe_allow_html=True)
with col4:
    tui = LOGOS_DIR / "Logo_Tui3.svg"
    if tui.exists():
        st.image(str(tui), width=100)

st.markdown("<p class='footer-note'>© 2025 Proyecto TFM - Redistribución Turística | Deep5</p>", unsafe_allow_html=True)
