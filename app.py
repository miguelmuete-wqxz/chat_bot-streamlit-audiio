import hashlib
import io
import os

import speech_recognition as sr
import streamlit as st
from anthropic import Anthropic
from gtts import gTTS

# ------------------------------------------------------------------
# 1. PERSONALIDAD DEL CHATBOT  (reemplaza con el resumen de tu grupo)
# ------------------------------------------------------------------
NOMBRE_BOT = "GrupoBot"

PERSONALIDAD = """
Eres GrupoBot, un chatbot que tiene la personalidad combinada de 5 compañeros de clase.

Datos del grupo (REEMPLAZA con tus entrevistas):
- Edades: entre 18 y 22 años. Mezcla de hombres y mujeres.
- Música: reguetón, pop y algo de rock.
- Películas: acción y comedia (Avengers, Rápidos y Furiosos).
- Deportes: fútbol principalmente; también ciclismo.
- Materia favorita: programación y matemáticas.
- Comida: pizza, hamburguesa y arepas.
- Fortalezas: trabajo en equipo, creatividad, puntualidad.
- Debilidades: procrastinación, se distraen con el celular.

Reglas:
- Habla en español, con tono juvenil, cercano y con humor.
- Responde en máximo 3 frases (tu respuesta se leerá en voz alta).
- Habla siempre en primera persona, como si esos gustos fueran tuyos.
- Si te preguntan algo fuera de tus gustos, responde según tu personalidad.
- No uses emojis ni markdown, porque se convertirá a voz.
"""

MODELO = "claude-haiku-4-5-20251001"  # cámbialo si usas otro modelo/proveedor
IDIOMA = "es"

# ------------------------------------------------------------------
# 2. FUNCIONES
# ------------------------------------------------------------------


def obtener_cliente() -> Anthropic | None:
    api_key = os.getenv("ANTHROPIC_API_KEY") or st.sidebar.text_input(
        "API key de Anthropic", type="password"
    )
    return Anthropic(api_key=api_key) if api_key else None


def transcribir(audio_bytes: bytes) -> str | None:
    """Voz -> texto (usa el reconocedor gratuito de Google)."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio, language="es-ES")
    except (sr.UnknownValueError, sr.RequestError):
        return None


def responder(cliente: Anthropic, historial: list[dict]) -> str:
    """Texto -> respuesta del LLM con la personalidad del grupo."""
    resp = cliente.messages.create(
        model=MODELO,
        max_tokens=300,
        system=PERSONALIDAD,
        messages=historial,
    )
    return resp.content[0].text


def texto_a_voz(texto: str) -> bytes:
    """Texto -> audio mp3 (gTTS)."""
    buffer = io.BytesIO()
    gTTS(text=texto, lang=IDIOMA).write_to_fp(buffer)
    return buffer.getvalue()


def procesar_mensaje(cliente: Anthropic, texto: str):
    st.session_state.mensajes.append({"role": "user", "content": texto})
    with st.spinner("Pensando..."):
        respuesta = responder(cliente, st.session_state.mensajes)
        st.session_state.mensajes.append({"role": "assistant", "content": respuesta})
        st.session_state.ultimo_audio = texto_a_voz(respuesta)


# ------------------------------------------------------------------
# 3. INTERFAZ STREAMLIT
# ------------------------------------------------------------------
st.set_page_config(page_title=NOMBRE_BOT, page_icon="🎙️")
st.title(f"🎙️ {NOMBRE_BOT}")
st.caption("Chatbot de voz con la personalidad de 5 compañeros de clase")

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []
    st.session_state.ultimo_audio = None
    st.session_state.audio_hash = None

cliente = obtener_cliente()
if st.sidebar.button("Reiniciar conversación"):
    st.session_state.mensajes = []
    st.session_state.ultimo_audio = None
    st.session_state.audio_hash = None
    st.rerun()

with st.sidebar.expander("Ver personalidad"):
    st.write(PERSONALIDAD)

if cliente is None:
    st.info("Ingresa tu API key en la barra lateral para comenzar.")
    st.stop()

# Historial
for m in st.session_state.mensajes:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# Entrada por voz
audio = st.audio_input("Presiona para hablar")
if audio is not None:
    audio_bytes = audio.getvalue()
    huella = hashlib.md5(audio_bytes).hexdigest()
    if huella != st.session_state.audio_hash:  # evita reprocesar el mismo audio
        st.session_state.audio_hash = huella
        texto = transcribir(audio_bytes)
        if texto:
            procesar_mensaje(cliente, texto)
            st.rerun()
        else:
            st.warning("No pude entender el audio, intenta de nuevo.")

# Entrada por texto (alternativa)
entrada = st.chat_input("...o escribe tu mensaje")
if entrada:
    procesar_mensaje(cliente, entrada)
    st.rerun()

# Reproducir la última respuesta
if st.session_state.ultimo_audio:
    st.audio(st.session_state.ultimo_audio, format="audio/mp3", autoplay=True)
