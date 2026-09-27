"""AI Vision Chatbot - a Streamlit app powered by Gemini Vision.

Run it with:  streamlit run app.py
"""

import io

import streamlit as st
from PIL import Image, ImageOps
from google import genai
from google.genai import types

from prompts import SYSTEM_PROMPT, DEFAULT_PROMPT

# The Gemini model that can look at images.
# Swap this for another vision model if you want to experiment.
MODEL_NAME = "gemini-2.5-flash"


# --- 1. Read the API key from .streamlit/secrets.toml -----------------
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error(
        "No API key found. Create a file called `.streamlit/secrets.toml` "
        'with: GEMINI_API_KEY = "your-key-here"'
    )
    st.stop()

if api_key == "YOUR_GEMINI_API_KEY":
    st.error("Open `.streamlit/secrets.toml` and replace the placeholder with your real Gemini API key.")
    st.stop()


# --- 2. Build the page ------------------------------------------------
st.set_page_config(page_title="MACROSNAP 🥗", page_icon="📸")
st.title("📸 MACROSNAP")
st.write("Upload an image, ask a question about it, and Gemini will answer.")


# --- 3. Let the user upload an image and type a question --------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
user_prompt = st.text_input("Ask a question about the image", value=DEFAULT_PROMPT)

if st.button("Ask MACROSNAP", disabled=uploaded_file is None):
    if not user_prompt.strip():
        st.warning("Please type a question first.")
        st.stop()

    with st.spinner("MACROSNAP is looking at the image..."):
        try:
            client = genai.Client(api_key=api_key)

            # Pillow prepares the image for the API: it fixes rotation from
            # phone cameras, turns transparent PNGs solid, and drops any
            # colour mode Gemini does not understand.
            image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")

            # Re-save as a plain JPEG and wrap the bytes in a Part.
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            image_part = types.Part.from_bytes(
                data=buffer.getvalue(),
                mime_type="image/jpeg",
            )

            # Send the image AND the question to Gemini.
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[image_part, user_prompt],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                ),
            )

            st.image(image, caption="The image you uploaded")
            st.markdown("### MACROSNAP's Response")
            st.write(response.text)

        except Exception as error:
            st.error(f"Something went wrong: {error}")
