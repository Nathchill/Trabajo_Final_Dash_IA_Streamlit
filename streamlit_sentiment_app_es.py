# streamlit_sentiment_app_es.py
# Versión en español del analizador de sentimientos con Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import io, os, re, base64
from collections import Counter
from nltk.corpus import stopwords
from nltk import bigrams
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
import nltk, spacy

# ---------------------------
# Configuración inicial
# ---------------------------
st.set_page_config(page_title="Análisis de Sentimientos en Español", layout="centered")

# ---------------------------
# Crear CSV simulado con opiniones en español
# ---------------------------
SAMPLE_CSV = "opiniones.csv"
SAMPLE_OPINIONS = [
    "El producto llegó a tiempo y funciona perfectamente, muy satisfecho.",
    "El servicio al cliente fue amable pero no resolvió mi problema.",
    "La batería dura muy poco, no recomiendo este artículo.",
    "Buena relación calidad-precio, cumple con lo prometido.",
    "Me encantó el empaque, se nota que cuidan los detalles.",
    "La aplicación se cierra sola a veces, necesita actualización.",
    "No tengo una opinión fuerte, hace lo que promete sin destacar.",
    "El envío fue rápido pero el producto tenía una abolladura.",
    "Excelente calidad de sonido y fácil de usar.",
    "Es muy caro para lo que ofrece, esperaba más rendimiento.",
    "Tuve que devolverlo, falló la primera semana.",
    "El soporte técnico resolvió mi caso en poco tiempo, excelente atención.",
    "Las instrucciones son confusas, me tomó mucho tiempo instalarlo.",
    "Es compacto y ligero, ideal para viajes.",
    "Lo compré como regalo y le encantó a la persona.",
    "Hace mucho ruido al funcionar, es molesto.",
    "La batería carga rápido y dura bastante.",
    "La cámara tiene una calidad mediocre comparada con otras marcas.",
    "La interfaz es intuitiva y fácil de navegar.",
    "Volvería a comprar de esta marca, fue una buena experiencia."
]

if not os.path.exists(SAMPLE_CSV):
    pd.DataFrame({"opinion": SAMPLE_OPINIONS}).to_csv(SAMPLE_CSV, index=False, encoding="utf-8")

# ---------------------------
# NLP setup
# ---------------------------
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nlp = spacy.load("es_core_news_sm")
except Exception:
    from spacy.cli import download
    download("es_core_news_sm")
    nlp = spacy.load("es_core_news_sm")

ES_STOPWORDS = set(stopwords.words("spanish"))
TOKEN_RE = re.compile(r"\b[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ']+\b")

def preprocess_text(text):
    text = text.lower()
    tokens = TOKEN_RE.findall(text)
    tokens = [t for t in tokens if t not in ES_STOPWORDS]
    doc = nlp(" ".join(tokens))
    lemmas = [token.lemma_ for token in doc if token.lemma_ != "-PRON-"]
    lemmas = [l for l in lemmas if len(l) > 1]
    return lemmas

def top_n_words(corpus, n=10):
    all_lemmas = []
    for t in corpus:
        all_lemmas.extend(preprocess_text(t))
    c = Counter(all_lemmas)
    return c.most_common(n)

def top_n_bigrams(corpus, n=10):
    all_bigrams = []
    for t in corpus:
        lemmas = preprocess_text(t)
        all_bigrams.extend([" ".join(bg) for bg in bigrams(lemmas)])
    c = Counter(all_bigrams)
    return c.most_common(n)

def make_wordcloud(corpus):
    all_text = " ".join([" ".join(preprocess_text(t)) for t in corpus])
    if not all_text.strip():
        all_text = "vacío"
    wc = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    return wc

# ---------------------------
# Modelo de sentimiento (HuggingFace en español)
# ---------------------------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")

sentiment_pipe = load_model()

def classify_list(texts):
    results = sentiment_pipe(texts)
    normalized = []
    for r in results:
        label = r.get("label", "").lower()
        if "pos" in label:
            normalized.append("positivo")
        elif "neg" in label:
            normalized.append("negativo")
        else:
            normalized.append("neutral")
    return normalized

# ---------------------------
# Interfaz Streamlit
# ---------------------------
st.title("📊 Análisis de Sentimientos (en Español)")
st.markdown("Sube un archivo CSV con una columna llamada `opinion`, o usa el ejemplo precargado.")

uploaded_file = st.file_uploader("Subir archivo CSV", type=["csv"])
use_sample = st.checkbox("Usar ejemplo precargado", value=True)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif use_sample:
    df = pd.read_csv(SAMPLE_CSV)
else:
    st.warning("Por favor, sube un archivo o usa el ejemplo.")
    st.stop()

if 'opinion' not in df.columns:
    df.rename(columns={df.columns[0]: 'opinion'}, inplace=True)

corpus = df['opinion'].astype(str).tolist()

# ---------------------------
# Nube de palabras
# ---------------------------
st.subheader("☁️ Nube de Palabras")
wc = make_wordcloud(corpus)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# ---------------------------
# Top palabras
# ---------------------------
st.subheader("🔝 Top 10 Palabras más Frecuentes")
top_words = top_n_words(corpus, 10)
if top_words:
    words, counts = zip(*top_words)
    fig, ax = plt.subplots()
    ax.bar(words, counts)
    ax.set_ylabel("Frecuencia")
    ax.set_xlabel("Palabra")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ---------------------------
# Top bigramas
# ---------------------------
st.subheader("🔠 Top 10 Bigramas más Comunes")
top_bgs = top_n_bigrams(corpus, 10)
if top_bgs:
    bgs, bg_counts = zip(*top_bgs)
    fig, ax = plt.subplots()
    ax.bar(bgs, bg_counts)
    ax.set_ylabel("Frecuencia")
    ax.set_xlabel("Bigrama")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ---------------------------
# Clasificación de sentimientos
# ---------------------------
st.subheader("💬 Resultados de Sentimiento")
sentiments = classify_list(corpus)
df["sentimiento"] = sentiments
st.dataframe(df)

# Gráfico circular
st.subheader("📈 Distribución de Sentimientos")
pie_data = df["sentimiento"].value_counts().reset_index()
pie_data.columns = ["Sentimiento", "Cantidad"]
fig, ax = plt.subplots()
ax.pie(pie_data["Cantidad"], labels=pie_data["Sentimiento"], autopct="%1.1f%%", startangle=90)
ax.axis("equal")
st.pyplot(fig)

# ---------------------------
# Clasificar nuevo comentario
# ---------------------------
st.subheader("✍️ Clasificar un nuevo comentario")
new_comment = st.text_area("Escribe una opinión nueva:")
if st.button("Clasificar"):
    if new_comment.strip():
        label = classify_list([new_comment.strip()])[0]
        mensajes = {
            'positivo': "🌟 ¡Gracias por tu comentario positivo! Nos alegra saber que estás satisfecho.",
            'neutral': "📝 Gracias por tu opinión. Tomaremos en cuenta tus observaciones.",
            'negativo': "⚠️ Lamentamos que tu experiencia no haya sido la mejor. Queremos mejorar y atender tu caso."
        }
        st.success(f"**Sentimiento detectado:** {label}")
        st.info(mensajes.get(label, "Gracias por tu comentario."))
    else:
        st.warning("Por favor escribe un comentario antes de clasificar.")
