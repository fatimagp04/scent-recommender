import os
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Recomendador Olfativo · TFG",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR   = Path(__file__).parent
COLLECTION = "dravnieks_descriptors"
GROQ_MODEL = "llama-3.3-70b-versatile"


def cargar_env(path: Path = BASE_DIR / ".env") -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


cargar_env()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()


# ── Cached resources ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando recursos (solo la primera vez)…")
def cargar_recursos():
    embeddings  = np.load(BASE_DIR / "descriptor_embeddings.npy")
    descriptors = pd.read_csv(BASE_DIR / "descriptor_list.csv")["descriptor"].tolist()
    behavior    = pd.read_csv(BASE_DIR / "dravnieks_behavior_named.csv", index_col=0)

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    qdrant = QdrantClient(":memory:")
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
    )
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(id=i, vector=embeddings[i].tolist(), payload={"descriptor": descriptors[i]})
            for i in range(len(descriptors))
        ],
    )
    return embeddings, descriptors, behavior, model, qdrant


# ── Pipeline ────────────────────────────────────────────────────────────────────
def extraer_atributos_con_pesos(texto: str, n: int, api_key: str) -> list[dict]:
    client = Groq(api_key=api_key, timeout=30.0)

    system_prompt = """You are an expert in sensory branding and olfactory marketing.
Analyze company identity texts and extract the core sensory/emotional attributes
that could be translated into olfactory experiences.

Rules:
- Output ONLY valid JSON, no other text, no markdown fences
- Each attribute must be in ENGLISH, 2-4 words, sensory and specific
- Attributes must be derived exclusively from the company text — never invent generic ones
- Good attributes: evoke a concrete smell, texture, temperature or natural element grounded in the brand's identity, ingredients, environment or values
- Bad attributes: abstract corporate concepts, generic buzzwords or anything not directly inferable from the text
- Weights must sum to exactly 1.0
- 'razon' explains briefly which part of the text inspired this attribute (in Spanish is fine)"""

    user_prompt = f"""Extract exactly {n} olfactory attributes with importance weights from this company text.

Company text:
{texto}

Return ONLY a JSON array (derive attributes from the text above, not placeholders):
[
  {{"atributo": "...", "peso": 0.XX, "razon": "..."}},
  ...
]"""

    raw = (
        client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        .choices[0]
        .message.content.strip()
    )
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        atributos = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if m:
            atributos = json.loads(m.group())
        else:
            raise ValueError(f"No se pudo parsear JSON:\n{raw}")

    total = sum(a["peso"] for a in atributos)
    for a in atributos:
        a["peso"] = round(a["peso"] / total, 4)
    return atributos


def buscar_descriptores(atributos: list[dict], top_k: int, embed_model, qdrant) -> list[dict]:
    vecs = embed_model.encode(
        [a["atributo"] for a in atributos], normalize_embeddings=True, show_progress_bar=False
    )
    resultado = []
    for i, atrib in enumerate(atributos):
        hits = qdrant.query_points(
            collection_name=COLLECTION, query=vecs[i].tolist(), limit=top_k
        ).points
        resultado.append(
            {
                **atrib,
                "descriptores": [
                    {"descriptor": h.payload["descriptor"], "coseno": round(h.score, 4)}
                    for h in hits
                ],
            }
        )
    return resultado


DESCRIPTORES_ES = {
    "FRUITY,CITRUS": "fruta cítrica",
    "LEMON": "limón",
    "GRAPEFRUIT": "pomelo",
    "ORANGE": "naranja",
    "FRUITY,OTHER THAN CITRUS": "fruta no cítrica",
    "PINEAPPLE": "piña",
    "GRAPE JUICE": "zumo de uva",
    "STRAWBERRY": "fresa",
    "APPLE, FRUIT": "manzana",
    "PEAR": "pera",
    "CANTALOUPE, HONEYDEW MELON": "melón",
    "PEACH FRUIT": "melocotón",
    "BANANA": "plátano",
    "FLORAL": "floral",
    "ROSE": "rosa",
    "VIOLETS": "violeta",
    "LAVENDER": "lavanda",
    "COLOGNE": "colonia",
    "MUSK": "almizcle",
    "PERFUMERY": "perfumería",
    "FRAGRANT": "fragante",
    "AROMATIC": "aromático",
    "HONEY": "miel",
    "CHERRY, BERRY": "cereza o frutos rojos",
    "ALMOND": "almendra",
    "NAIL POLISH REMOVER": "quitaesmalte",
    "NUTTY, WALNUT ETC.": "frutos secos",
    "SPICY": "especiado",
    "CLOVE": "clavo",
    "CINNAMON": "canela",
    "LAUREL LEAVES": "laurel",
    "TEA LEAVES": "hojas de té",
    "SEASONING FOR MEAT": "condimento para carne",
    "BLACK PEPPER": "pimienta negra",
    "GREEN PEPPER": "pimiento verde",
    "DILL": "eneldo",
    "CARAWAY": "alcaravea",
    "OAK WOOD,COGNAC": "madera de roble o coñac",
    "WOODY, RESINOUS": "madera y resina",
    "CEDARWOOD": "madera de cedro",
    "MOTHBALLS": "naftalina",
    "MINTY, PEPPERMINT": "menta",
    "CAMPHOR": "alcanfor",
    "EUCALYPTUS": "eucalipto",
    "CHOCOLATE": "chocolate",
    "VANILLA": "vainilla",
    "SWEET": "dulce",
    "MAPLE SYRUP": "sirope de arce",
    "CARAMEL": "caramelo",
    "MALTY": "malta",
    "RAISINS": "pasas",
    "MOLASSES": "melaza",
    "COCONUT": "coco",
    "ANISE, LICORICE": "anís o regaliz",
    "ALCOHOLIC": "alcohólico",
    "ETHERISH, ANAESTHETIC": "éter o anestésico",
    "CLEANING FLUID": "producto de limpieza",
    "GASOLINE, SOLVENT": "gasolina o disolvente",
    "TURPENTINE, PINE OIL": "trementina o aceite de pino",
    "GERANIUM LEAVES": "hojas de geranio",
    "CELERY": "apio",
    "FRESH GREEN VEGETABLES": "verduras verdes frescas",
    "CRUSHED WEEDS": "hierbas machacadas",
    "CRUSHED GRASS": "hierba cortada",
    "HERBAL, GREEN,CUTGRASS": "herbal y verde",
    "RAW CUCUMBER": "pepino",
    "HAY": "heno",
    "GRAINY (AS GRAIN)": "cereal",
    "YEASTY": "levadura",
    "BAKERY (FRESH BREAD)": "pan recién hecho",
    "SOUR MILK": "leche agria",
    "FERMENTED, ROTTEN FRUIT": "fruta fermentada",
    "BEERY": "cerveza",
    "SOAPY": "jabón",
    "LEATHER": "cuero",
    "CARDBOARD": "cartón",
    "ROPE": "cuerda",
    "WET PAPER": "papel mojado",
    "WET WOOL,WET DOG": "lana mojada",
    "DIRTY LINEN": "ropa sucia",
    "STALE": "rancio o cerrado",
    "MUSTY, EARTHY, MOLDY": "humedad, tierra y moho",
    "RAW POTATO": "patata cruda",
    "MOUSE": "olor animal pequeño",
    "MUSHROOM": "seta",
    "PEANUT BUTTER": "mantequilla de cacahuete",
    "BEANY": "legumbre",
    "EGGY, FRESH EGGS": "huevo",
    "BARK,BIRCHBARK": "corteza",
    "CORK": "corcho",
    "BURNT,SMOKY": "quemado y ahumado",
    "FRESH TOBACCO SMOKE": "humo de tabaco fresco",
    "INCENSE": "incienso",
    "COFFEE": "café",
    "STALE TOBACCO SMOKE": "humo de tabaco viejo",
    "BURNT PAPER": "papel quemado",
    "BURNT MILK": "leche quemada",
    "BURNT RUBBER": "goma quemada",
    "TAR": "alquitrán",
    "CREOSOTE": "creosota",
    "DISINFECTANT, CARBOLIC": "desinfectante",
    "MEDICINAL": "medicinal",
    "CHEMICAL": "químico",
    "BITTER": "amargo",
    "SHARP, PUNGENT, ACID": "punzante y ácido",
    "SOUR, VINEGAR": "vinagre",
    "SAUERKRAUT": "col fermentada",
    "AMMONIA": "amoníaco",
    "URINE": "orina",
    "CAT URINE": "orina de gato",
    "FISHY": "pescado",
    "KIPPERY,SMOKED FISH": "pescado ahumado",
    "SEMINAL, SPERM-LIKE": "seminal",
    "NEW RUBBER": "goma nueva",
    "SOOTY": "hollín",
    "BURNT CANDLE": "vela quemada",
    "KEROSENE": "queroseno",
    "OILY, FATTY": "aceitoso o graso",
    "BUTTERY, FRESH BUTTER": "mantequilla",
    "PAINT": "pintura",
    "VARNISH": "barniz",
    "POPCORN": "palomitas",
    "FRIED CHICKEN": "pollo frito",
    "MEATY ( COOKED,GOOD)": "carne cocinada",
    "SOUPY": "caldo",
    "COOKED VEGETABLES": "verduras cocinadas",
    "RANCID": "rancio",
    "SWEATY": "sudor",
    "CHEESY": "queso",
    "HOUSEHOLD GAS": "gas doméstico",
    "SULFIDIC": "azufrado",
    "GARLIC, ONION": "ajo o cebolla",
    "METALLIC": "metálico",
    "BLOOD, RAW MEAT": "sangre o carne cruda",
    "ANIMAL": "animal",
    "SEWER": "alcantarilla",
    "PUTRID, FOUL, DECAYED": "podrido",
    "FECAL, LIKE MANURE": "estiércol",
    "CADAVEROUS": "cadavérico",
    "SICKENING": "nauseabundo",
    "DRY, POWDERY": "seco y empolvado",
    "CHALKY": "tiza",
    "LIGHT": "ligero",
    "HEAVY": "pesado",
    "COOL,COOLING": "fresco",
    "WARM": "cálido",
}


def traducir_descriptor(descriptor: str) -> str:
    return DESCRIPTORES_ES.get(descriptor, descriptor.lower().replace(",", ", "))


OLORES_DEMASIADO_GENERICOS = {
    "fragante",
    "perfumería",
    "aromático",
    "dulce",
    "ligero",
    "pesado",
    "fresco",
    "cálido",
}


def olores_tipicos_molecula(fila: pd.Series, descriptors: list[str], limite: int = 5) -> list[str]:
    ordenados = fila[descriptors].sort_values(ascending=False)
    concretos = []
    genericos = []

    for desc, valor in ordenados.items():
        if valor <= 0:
            continue

        olor = traducir_descriptor(desc)
        destino = genericos if olor in OLORES_DEMASIADO_GENERICOS else concretos
        if olor not in concretos and olor not in genericos:
            destino.append(olor)

        if len(concretos) >= limite:
            break

    return (concretos + genericos)[:limite]


def calcular_scores_moleculas(
    atributos_con_descriptores: list[dict], behavior: pd.DataFrame, descriptors: list[str]
) -> pd.DataFrame:
    n_mols         = len(behavior)
    mol_names      = behavior["molecule_name"].values
    scores_totales = np.zeros(n_mols)
    desglose       = defaultdict(list)

    for atrib in atributos_con_descriptores:
        peso = atrib["peso"]
        for d in atrib["descriptores"]:
            desc, coseno = d["descriptor"], d["coseno"]
            if desc not in behavior.columns:
                continue
            s = behavior[desc].values
            contrib = peso * coseno * s / 100
            scores_totales += contrib
            for idx, c in enumerate(contrib):
                if c > 0.001:
                    desglose[idx].append(
                        {
                            "atributo":        atrib["atributo"],
                            "peso_atributo":   peso,
                            "descriptor":      desc,
                            "coseno":          coseno,
                            "score_dravnieks": round(float(s[idx]), 1),
                            "contribucion":    round(float(c), 5),
                        }
                    )

    rows = [
        {
            "molecule_name": mol_names[i],
            "score_total":   round(float(scores_totales[i]), 5),
            "desglose":      sorted(desglose[i], key=lambda x: -x["contribucion"]),
            "olores_comunes": olores_tipicos_molecula(behavior.iloc[i], descriptors),
        }
        for i in range(n_mols)
    ]
    return pd.DataFrame(rows).sort_values("score_total", ascending=False).reset_index(drop=True)


def generar_explicacion(
    texto: str, atributos: list[dict], resultados: pd.DataFrame, top_n: int, api_key: str
) -> str:
    client   = Groq(api_key=api_key, timeout=30.0)
    top_mols = resultados.head(top_n)

    resumen_mols = [
        f"- {r['molecule_name']} (score={r['score_total']:.4f}, "
        f"descriptores: {', '.join(c['descriptor'] for c in r['desglose'][:3])})"
        for _, r in top_mols.iterrows()
    ]
    resumen_attrs = [f"- '{a['atributo']}' (peso={a['peso']:.2f}): {a['razon']}" for a in atributos]

    prompt = f"""Eres un experto en marketing olfativo y branding sensorial.

Empresa analizada:
{texto.strip()}

Atributos sensoriales identificados:
{chr(10).join(resumen_attrs)}

Moléculas olfativas recomendadas (ranking):
{chr(10).join(resumen_mols)}

Escribe un análisis claro y académico (máx. 250 palabras) explicando:
1. Por qué estas moléculas representan la identidad de la empresa
2. Cómo cada molécula conecta con los atributos identificados
3. Qué experiencia olfativa conjunta transmiten al consumidor

Tono: riguroso pero accesible. En español."""

    return (
        client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=500,
        )
        .choices[0]
        .message.content.strip()
    )


# ── Charts ──────────────────────────────────────────────────────────────────────
def fig_atributos_bar(atributos: list[dict]):
    df = pd.DataFrame(atributos).sort_values("peso")
    fig = px.bar(
        df, x="peso", y="atributo", orientation="h",
        color="peso", color_continuous_scale="Teal",
        text=df["peso"].apply(lambda x: f"{x:.0%}"),
        labels={"peso": "Peso", "atributo": ""},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=0, r=50, t=10, b=0),
        height=max(220, len(atributos) * 52),
    )
    return fig


def fig_cosine_heatmap(atribs_desc: list[dict]):
    attrs     = [a["atributo"] for a in atribs_desc]
    all_descs = list({d["descriptor"] for a in atribs_desc for d in a["descriptores"]})

    matrix = np.zeros((len(attrs), len(all_descs)))
    for i, a in enumerate(atribs_desc):
        for d in a["descriptores"]:
            j = all_descs.index(d["descriptor"])
            matrix[i, j] = d["coseno"]

    fig = go.Figure(
        go.Heatmap(
            z=matrix, x=all_descs, y=attrs,
            colorscale="Teal", zmin=0, zmax=1,
            text=np.round(matrix, 2), texttemplate="%{text}",
            hoverongaps=False,
        )
    )
    fig.update_layout(
        xaxis_tickangle=-40,
        margin=dict(l=0, r=0, t=10, b=80),
        height=max(280, len(attrs) * 58),
    )
    return fig


def fig_score_histogram(resultados: pd.DataFrame, corte_score: float, top_n: int):
    fig = px.histogram(
        resultados, x="score_total", nbins=30,
        color_discrete_sequence=["#009688"],
        labels={"score_total": "Score", "count": "N"},
    )
    fig.add_vline(
        x=corte_score, line_dash="dash", line_color="orange",
        annotation_text=f"Top-{top_n}", annotation_position="top right",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=260, bargap=0.05)
    return fig


def _label_sankey(texto: str, limite_linea: int = 24, limite_total: int = 52) -> str:
    texto = str(texto).strip()
    if len(texto) > limite_total:
        texto = texto[: limite_total - 1].rstrip() + "..."

    lineas = []
    linea_actual = ""
    for palabra in texto.split():
        candidato = f"{linea_actual} {palabra}".strip()
        if len(candidato) <= limite_linea:
            linea_actual = candidato
        else:
            if linea_actual:
                lineas.append(linea_actual)
            linea_actual = palabra
    if linea_actual:
        lineas.append(linea_actual)

    return "<br>".join(lineas)


def fig_flujo_atributo_descriptor_molecula(
    atribs_desc: list[dict], resultados: pd.DataFrame, top_n: int
):
    top = resultados.head(top_n).copy()

    descriptor_molecula = defaultdict(float)
    atributo_descriptor = defaultdict(float)

    for _, row in top.iterrows():
        score_total = float(row["score_total"]) or 1.0
        for contrib in row["desglose"]:
            valor = float(contrib["contribucion"])
            if valor <= 0:
                continue

            descriptor = contrib["descriptor"]
            molecule = row["molecule_name"]
            descriptor_molecula[(descriptor, molecule)] += valor / score_total

    for atrib in atribs_desc:
        atributo = atrib["atributo"]
        for desc in atrib["descriptores"]:
            descriptor = desc["descriptor"]
            atributo_descriptor[(atributo, descriptor)] += float(atrib["peso"]) * float(desc["coseno"])

    nodos = []
    node_index = {}

    def add_node(nombre: str) -> int:
        if nombre not in node_index:
            node_index[nombre] = len(nodos)
            nodos.append(nombre)
        return node_index[nombre]

    fuentes, destinos, valores, link_colors = [], [], [], []

    for (atributo, descriptor), valor in sorted(atributo_descriptor.items()):
        fuentes.append(add_node(atributo))
        destinos.append(add_node(descriptor))
        valores.append(valor)
        link_colors.append("rgba(0, 150, 136, 0.24)")

    for (descriptor, molecule), valor in sorted(descriptor_molecula.items()):
        fuentes.append(add_node(descriptor))
        destinos.append(add_node(molecule))
        valores.append(valor)
        link_colors.append("rgba(216, 117, 75, 0.24)")

    for molecule in top["molecule_name"]:
        add_node(molecule)

    atributos = {a["atributo"] for a in atribs_desc}
    moleculas = set(top["molecule_name"])
    colores = []
    for nodo in nodos:
        if nodo in atributos:
            colores.append("#17998f")
        elif nodo in moleculas:
            colores.append("#d8754b")
        else:
            colores.append("#79c5c2")

    grupos = {
        "atributos": [n for n in nodos if n in atributos],
        "descriptores": [n for n in nodos if n not in atributos and n not in moleculas],
        "moleculas": [n for n in nodos if n in moleculas],
    }
    posiciones_x = []
    posiciones_y = []
    for nodo in nodos:
        if nodo in grupos["atributos"]:
            grupo = grupos["atributos"]
            posiciones_x.append(0.08)
        elif nodo in grupos["moleculas"]:
            grupo = grupos["moleculas"]
            posiciones_x.append(0.92)
        else:
            grupo = grupos["descriptores"]
            posiciones_x.append(0.50)

        posiciones_y.append((grupo.index(nodo) + 1) / (len(grupo) + 1))

    anotaciones = []
    for i, nodo in enumerate(nodos):
        if nodo in grupos["atributos"]:
            x_label = 0.005
            xanchor = "left"
            align = "left"
        elif nodo in grupos["moleculas"]:
            x_label = 0.995
            xanchor = "right"
            align = "right"
        else:
            x_label = 0.525
            xanchor = "left"
            align = "left"

        anotaciones.append(
            dict(
                x=x_label,
                y=1 - posiciones_y[i],
                xref="paper",
                yref="paper",
                text=_label_sankey(nodo, limite_linea=22, limite_total=48),
                showarrow=False,
                xanchor=xanchor,
                yanchor="middle",
                align=align,
                font=dict(size=12, color="#000000", family="Arial, sans-serif"),
                bgcolor="rgba(255, 255, 255, 0.94)",
                bordercolor="rgba(255, 255, 255, 0)",
                borderpad=2,
            )
        )

    fig = go.Figure(
        go.Sankey(
            arrangement="fixed",
            node=dict(
                pad=8,
                thickness=15,
                line=dict(color="rgba(25, 35, 45, 0.40)", width=0.8),
                label=["" for _ in nodos],
                customdata=nodos,
                color=colores,
                x=posiciones_x,
                y=posiciones_y,
                hovertemplate="%{customdata}<extra></extra>",
            ),
            link=dict(
                source=fuentes,
                target=destinos,
                value=valores,
                color=link_colors,
                hovertemplate="%{source.label} -> %{target.label}<br>Peso relativo: %{value:.3f}<extra></extra>",
            ),
        )
    )
    fig.update_layout(
        title=dict(
            text="Diagrama de flujo Atributo -> Descriptor -> Molecula",
            x=0.01,
            font=dict(size=20, color="#172033", family="Arial, sans-serif"),
        ),
        font=dict(size=11, color="#111111", family="Arial, sans-serif"),
        height=max(540, 20 * max(1, len(nodos))),
        margin=dict(l=95, r=95, t=82, b=38),
        paper_bgcolor="white",
        plot_bgcolor="white",
        annotations=anotaciones,
    )
    return fig


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parámetros")
    api_key = GROQ_API_KEY

    st.divider()
    n_atributos = st.slider("Atributos a extraer",       min_value=3, max_value=8, value=5)
    top_k_desc  = st.slider("Top-K descriptores / attr", min_value=2, max_value=8, value=4)
    top_n_mols  = st.slider("Moléculas en leaderboard",  min_value=3, max_value=15, value=8)

    st.divider()

# ── Header ───────────────────────────────────────────────────────────────────────
st.title("Recomendador de Identidad Olfativa")

# ── Input ────────────────────────────────────────────────────────────────────────
texto_empresa = st.text_area(
    "Texto de la empresa",
    height=180,
    placeholder=(
        "Pega aquí la misión, visión y valores de la empresa…\n\n"
        "Ejemplo: «Somos una marca de cosmética natural fundada en Galicia. "
        "Nuestra misión es reconectar a las personas con la naturaleza a través "
        "de ingredientes atlánticos puros: algas marinas, líquenes de montaña, "
        "aceites de camelia…»"
    ),
)

_, btn_col, _ = st.columns([3, 2, 3])
with btn_col:
    analizar = st.button("Analizar", type="primary", use_container_width=True)

# ── Run pipeline ─────────────────────────────────────────────────────────────────
if analizar:
    if not texto_empresa.strip():
        st.warning("Introduce el texto de la empresa antes de analizar.")
        st.stop()
    if not api_key:
        st.error("No se encontró GROQ_API_KEY. Revisa el archivo .env en la carpeta del proyecto.")
        st.stop()

    _, descriptors, behavior, embed_model, qdrant = cargar_recursos()

    steps = st.status("Ejecutando pipeline…", expanded=True)
    with steps:
        st.write("Extrayendo atributos sensoriales con Groq LLM…")
        atributos = extraer_atributos_con_pesos(texto_empresa, n_atributos, api_key)

        st.write("Calculando similitud coseno contra descriptores Dravnieks…")
        atribs_desc = buscar_descriptores(atributos, top_k_desc, embed_model, qdrant)

        st.write("Calculando scores ponderados para las 160 moléculas…")
        resultados = calcular_scores_moleculas(atribs_desc, behavior, descriptors)

        st.write("Generando explicación en lenguaje natural…")
        explicacion = generar_explicacion(
            texto_empresa, atributos, resultados, min(5, top_n_mols), api_key
        )
    steps.update(label="Análisis completado", state="complete", expanded=False)

    # store in session so reruns no re-ejecuten el pipeline
    st.session_state["resultado"] = dict(
        atributos=atributos,
        atribs_desc=atribs_desc,
        resultados=resultados,
        explicacion=explicacion,
    )

# ── Results ───────────────────────────────────────────────────────────────────────
if "resultado" in st.session_state:
    r           = st.session_state["resultado"]
    atributos   = r["atributos"]
    atribs_desc = r["atribs_desc"]
    resultados  = r["resultados"]
    explicacion = r["explicacion"]
    top         = resultados.head(top_n_mols)
    score_max   = top.iloc[0]["score_total"]
    corte_score = top.iloc[-1]["score_total"]

    tab1, tab2, tab3 = st.tabs(["Atributos", "Moléculas elegidas", "Conclusión"])

    # ── TAB 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Pesos de los atributos extraídos")
            st.plotly_chart(fig_atributos_bar(atributos), use_container_width=True)

        with col_b:
            st.subheader("Tabla de atributos")
            df_a = pd.DataFrame(atributos)[["atributo", "peso", "razon"]].copy()
            df_a["peso"] = df_a["peso"].apply(lambda x: f"{x:.0%}")
            st.dataframe(
                df_a, use_container_width=True, hide_index=True,
                column_config={
                    "atributo": st.column_config.TextColumn("Atributo", width="medium"),
                    "peso":     st.column_config.TextColumn("Peso",     width="small"),
                    "razon":    st.column_config.TextColumn("Razón derivada del texto"),
                },
            )

        st.divider()
        st.subheader("Similitud coseno: Atributos × Descriptores Dravnieks")
        st.plotly_chart(fig_cosine_heatmap(atribs_desc), use_container_width=True)

    # ── TAB 2 ─────────────────────────────────────────────────────────────────
    with tab2:
        st.subheader(f"Top {top_n_mols} moléculas recomendadas")

        for rank, row in top.iterrows():
            contribs_principales = row["desglose"][:3]
            atributos_activadores = ", ".join(
                dict.fromkeys(c["atributo"] for c in contribs_principales)
            )
            descriptores_usados = ", ".join(
                dict.fromkeys(c["descriptor"] for c in contribs_principales)
            )
            olores_comunes = row.get("olores_comunes", [])
            if not olores_comunes:
                olores_comunes = list(dict.fromkeys(traducir_descriptor(c["descriptor"]) for c in contribs_principales))
            olores_comunes_txt = ", ".join(olores_comunes)
            aportes = "; ".join(
                f"{c['atributo']} con {c['descriptor']}: "
                f"{(c['contribucion'] / row['score_total']):.0%} del score"
                for c in contribs_principales
            )

            with st.container(border=True):
                st.markdown(f"### #{rank + 1} {row['molecule_name']}")
                st.markdown(f"**Score:** {row['score_total']:.4f}")
                st.markdown(
                    f"- **Atributos que la activaron:** {atributos_activadores}\n"
                    f"- **Descriptores usados:** {descriptores_usados}\n"
                    f"- **Cuánto contribuyó cada uno:** {aportes}\n"
                    f"- **Asociada típicamente a:** {olores_comunes_txt}"
                )

    # ── TAB 3 ─────────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Conclusión")
        st.markdown(explicacion)

        st.divider()
        st.subheader("Distribución de las 160 moléculas")
        st.plotly_chart(
            fig_score_histogram(resultados, corte_score, top_n_mols),
            use_container_width=True,
        )

        st.divider()
        st.plotly_chart(
            fig_flujo_atributo_descriptor_molecula(atribs_desc, resultados, top_n_mols),
            use_container_width=True,
        )
