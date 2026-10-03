"""MacroSnap - AI-powered nutrition tracking.

Snap a photo of your meal, chat about calories & macros, and text the
daily summary to yourself on WhatsApp.

Run it with:  streamlit run app.py
"""

import hashlib
import re

import streamlit as st
from google import genai
from google.genai import types
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

from prompts import SYSTEM_PROMPT, SUMMARY_REQUEST_PROMPT

# The Gemini model that can look at images.
MODEL_NAME = "gemini-3.5-flash"

# Asked whenever the user uploads a photo without typing a question.
IMAGE_ONLY_PROMPT = "What is this meal? Give me the calories and macros."

# WhatsApp template variables are capped, so the summary is trimmed to fit.
WHATSAPP_MAX_CHARS = 1500


# --- 1. Build the page ------------------------------------------------
# set_page_config has to be the first Streamlit command in the script.
st.set_page_config(page_title="MacroSnap", page_icon="🥗", layout="centered")

# Inject custom CSS for a polished dark theme with a card-based layout.
st.markdown(
    """
    <style>
    /* ---------- Base dark theme ---------- */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    .block-container {
        max-width: 820px;
        padding-top: 0.5rem;
        padding-bottom: 3rem;
    }

    /* ---------- Typography ---------- */
    h1, h2, h3 { color: #e6edf3; }
    p { color: #8b949e; }

    /* ---------- Cards ---------- */
    .card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 18px;
    }

    /* ---------- Primary button ---------- */
    .stButton > button {
        background-color: #238636;
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 15px;
        transition: background-color 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #2ea043;
        color: #ffffff;
    }

    /* ---------- Chat input ---------- */
    .stChatInput > div {
        border-radius: 12px;
        border: 1px solid #30363d;
    }

    /* ---------- File uploader ---------- */
    .stFileUploader {
        border: 2px dashed #30363d;
        border-radius: 12px;
        padding: 20px;
    }

    /* ---------- Sidebar ---------- */
    .stSidebar {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }

    /* ---------- Stat cards ---------- */
    .stat-card {
        background-color: #21262d;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 12px;
        text-align: center;
    }

    /* ---------- Hide default Streamlit header/footer ---------- */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- 2. Secrets --------------------------------------------------------
# The Gemini key is required; the Twilio credentials are only needed when the
# user sends a summary, so they are checked when that button is pressed.
try:
    st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error(
        "No Gemini API key found. Create a file called "
        '`.streamlit/secrets.toml` with:\n\nGEMINI_API_KEY = "your-key-here"'
    )
    st.stop()


# --- 3. Cached API clients ---------------------------------------------
@st.cache_resource
def get_gemini_client():
    """The Gemini client, created once per session from the stored API key."""
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


@st.cache_resource
def get_twilio_client():
    """The Twilio client, created once per session from the stored secrets."""
    return Client(
        st.secrets["TWILIO_ACCOUNT_SID"],
        st.secrets["TWILIO_AUTH_TOKEN"],
    )


# --- 4. Helper functions (defined BEFORE use) ---------------------------
def clean_whatsapp_text(text):
    """Convert a Gemini response into clean WhatsApp-compatible plain text.

    Removes Markdown formatting, collapses whitespace, and caps the length.
    """
    if not text:
        return ""
    # Strip Markdown bold/italic markers and headers.
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Collapse whitespace/newlines into single spaces.
    collapsed = re.sub(r"\s+", " ", text).strip()
    return collapsed[:WHATSAPP_MAX_CHARS]


def send_whatsapp(to_number, user_name, summary):
    """Send via Twilio's WhatsApp Content Template (current trial version).

    Current trial template is Event Notifications with zero variables, so the
    working request sends only the ContentSid - no ContentVariables, no body.
    The ``user_name`` / ``summary`` args are kept so the caller (Gemini
    summary generation + clean_whatsapp_text) stays intact and a future
    two-variable template can reuse them without changing call sites.

    Returns (sent, detail): the message SID on success, or a safe error
    message on failure. Credentials are never included in the error text.
    """
    raw_to = str(to_number).strip()
    if raw_to.lower().startswith("whatsapp:"):
        raw_to = raw_to[len("whatsapp:"):]
    formatted_number = f"whatsapp:{raw_to}"

    # All secrets come from .streamlit/secrets.toml - nothing hardcoded.
    sender = str(st.secrets["TWILIO_WHATSAPP_FROM"]).strip()
    content_sid = str(st.secrets["TWILIO_CONTENT_SID"]).strip()

    try:
        client = get_twilio_client()
        message = client.messages.create(
            from_=sender,
            to=formatted_number,
            content_sid=content_sid,
        )
        return True, message.sid
    except TwilioRestException as e:
        # Surface the ACTUAL Twilio error (code / message / HTTP status).
        # Never include Account SID, Auth Token, or API keys here.
        code = getattr(e, "code", None)
        msg = getattr(e, "msg", None) or str(e)
        status = getattr(e, "status", None)
        twilio_details = getattr(e, "details", None)
        detail = f"Twilio error {code}: {msg} | HTTP status: {status}"
        if twilio_details:
            detail += f" | Details: {twilio_details}"
        return False, detail
    except Exception as error:
        return False, f"Could not send the WhatsApp message: {error}"


def parse_nutrition(text):
    """Extract calories, protein, carbs, and fat from an AI response.

    Returns a dict with keys 'calories', 'protein', 'carbs', 'fat' and
    string values like '520 kcal', '62 g', etc. Values not found are omitted.
    """
    if not text:
        return {}

    results = {}

    # Calories: "520 kcal", "520 calories", "Calories: 520", etc.
    cal_match = re.search(r"(\d[\d,]*)\s*(?:kcal|calories|cal)\b", text, re.IGNORECASE)
    if cal_match:
        results["calories"] = f"{cal_match.group(1)} kcal"

    # Protein: "62g protein", "Protein: 62g", "62 g protein", etc.
    protein_match = re.search(r"(\d[\d,]*)\s*g\s*(?:of\s+)?protein", text, re.IGNORECASE)
    if not protein_match:
        protein_match = re.search(r"protein[:\s]+(\d[\d,]*)\s*g", text, re.IGNORECASE)
    if protein_match:
        results["protein"] = f"{protein_match.group(1)} g"

    # Carbs
    carbs_match = re.search(r"(\d[\d,]*)\s*g\s*(?:of\s+)?(?:carbs|carbohydrates)", text, re.IGNORECASE)
    if not carbs_match:
        carbs_match = re.search(r"(?:carbs|carbohydrates)[:\s]+(\d[\d,]*)\s*g", text, re.IGNORECASE)
    if carbs_match:
        results["carbs"] = f"{carbs_match.group(1)} g"

    # Fat
    fat_match = re.search(r"(\d[\d,]*)\s*g\s*(?:of\s+)?fat", text, re.IGNORECASE)
    if not fat_match:
        fat_match = re.search(r"fat[:\s]+(\d[\d,]*)\s*g", text, re.IGNORECASE)
    if fat_match:
        results["fat"] = f"{fat_match.group(1)} g"

    return results


def create_chat():
    """Start a Gemini chat with MacroSnap's system prompt."""
    return get_gemini_client().chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )


# --- 5. Onboarding -----------------------------------------------------
# st.session_state keeps these values for the rest of the browser session,
# so the form only appears once.
if "onboarded" not in st.session_state:
    st.session_state["onboarded"] = False

if not st.session_state["onboarded"]:
    # Polished welcome card instead of a default Streamlit form.
    st.markdown(
        """
        <div class="card" style="max-width: 480px; margin: 4rem auto; text-align: center;">
            <h1 style="margin-bottom: 4px;">MacroSnap</h1>
            <p style="margin-bottom: 24px;">Your personal AI nutrition assistant.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("onboarding_form"):
        name = st.text_input("Name", placeholder="Your name")
        whatsapp_number = st.text_input(
            "WhatsApp number",
            placeholder="+1234567890",
            help="Include your country code. MacroSnap will text you here.",
        )
        submitted = st.form_submit_button("Continue")

    if submitted:
        if not name.strip():
            st.error("Please enter your name.")
        elif not whatsapp_number.strip():
            st.error("Please enter your WhatsApp number.")
        else:
            st.session_state["name"] = name.strip()
            st.session_state["whatsapp_number"] = whatsapp_number.strip()
            st.session_state["messages"] = []
            st.session_state["chat"] = create_chat()
            st.session_state["onboarded"] = True
            # Re-run the script so the app below shows up right away.
            st.rerun()

    # Nothing else to do until the user fills in the form.
    st.stop()


# --- 6. Header ----------------------------------------------------------
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 20px;">
        <div>
            <span style="font-size: 22px; font-weight: 700; color: #e6edf3;">MacroSnap</span>
            <span style="font-size: 13px; color: #8b949e; margin-left: 12px;">AI-powered nutrition tracking</span>
        </div>
        <span style="font-size: 12px; color: #8b949e;">Your AI Nutrition Assistant</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# --- 7. Sidebar ----------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding: 16px 0;">
            <span style="font-size: 16px; font-weight: 600; color: #e6edf3;">MacroSnap</span>
            <p style="font-size: 12px; color: #8b949e; margin: 4px 0 16px 0;">AI Nutrition Assistant</p>
            <p style="font-size: 13px; color: #e6edf3; margin: 0;">{st.session_state['name']}</p>
            <p style="font-size: 12px; color: #8b949e; margin: 4px 0 0 0;">{st.session_state['whatsapp_number']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # WhatsApp connection status
    twilio_ready = all(
        st.secrets.get(key)
        for key in (
            "TWILIO_ACCOUNT_SID",
            "TWILIO_AUTH_TOKEN",
            "TWILIO_WHATSAPP_FROM",
            "TWILIO_CONTENT_SID",
        )
    )
    status_color = "#238636" if twilio_ready else "#d29922"
    status_text = "Connected" if twilio_ready else "Not configured"
    st.markdown(
        f"""
        <div style="padding: 12px 0; border-top: 1px solid #30363d;">
            <p style="font-size: 11px; color: #8b949e; margin: 0 0 4px 0;">WhatsApp</p>
            <p style="font-size: 13px; color: {status_color}; margin: 0;">{status_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Today's totals
    totals = st.session_state.get("daily_totals", {})
    if totals:
        st.markdown(
            """
            <div style="padding: 12px 0; border-top: 1px solid #30363d;">
                <p style="font-size: 11px; color: #8b949e; margin: 0 0 8px 0;">Today</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cols = st.columns(4)
        for col, (label, key) in zip(cols, [("Cal", "calories"), ("Pro", "protein"), ("Carb", "carbs"), ("Fat", "fat")]):
            with col:
                val = totals.get(key, 0)
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <p style="font-size: 11px; color: #8b949e; margin: 0;">{label}</p>
                        <p style="font-size: 16px; font-weight: 600; color: #e6edf3; margin: 4px 0 0 0;">{val}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Reset / New meal
    if st.button("New meal", key="reset_button"):
        st.session_state["messages"] = []
        st.session_state["last_image_fingerprint"] = None
        st.session_state["daily_totals"] = {}
        st.rerun()


# --- 8. Meal Analysis card ----------------------------------------------
st.markdown(
    """
    <div class="card">
        <h3 style="margin-top: 0;">Meal Analysis</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload your meal",
    type=["jpg", "jpeg", "png"],
    help="Take a photo or choose an image to estimate calories and macros",
)

# Show image preview inside the card
if uploaded_file is not None:
    st.image(uploaded_file, use_container_width=True)


# --- 9. Chat history -----------------------------------------------------
for message in st.session_state.get("messages", []):
    if message["kind"] == "image":
        continue  # Images are shown in the Meal Analysis card
    role = message["role"]
    content = message["content"]
    if role == "user":
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
                <div style="background-color: #238636; color: #ffffff; border-radius: 16px 16px 4px 16px; padding: 12px 16px; max-width: 75%;">
                    <p style="margin: 0; font-size: 14px;">{content}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Assistant message with optional nutrition stats
        nutrition = parse_nutrition(content)
        if nutrition:
            stat_cards = "".join(
                f'<div class="stat-card"><p style="font-size: 11px; color: #8b949e; margin: 0;">{label}</p>'
                f'<p style="font-size: 18px; font-weight: 600; color: #e6edf3; margin: 4px 0 0 0;">{value}</p></div>'
                for label, value in [
                    ("Calories", nutrition.get("calories", "—")),
                    ("Protein", nutrition.get("protein", "—")),
                    ("Carbs", nutrition.get("carbs", "—")),
                    ("Fat", nutrition.get("fat", "—")),
                ]
            )
            stats_html = f'<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 12px 0;">{stat_cards}</div>'
        else:
            stats_html = ""

        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-start; margin-bottom: 12px;">
                <div style="background-color: #21262d; border: 1px solid #30363d; border-radius: 16px 16px 16px 4px; padding: 12px 16px; max-width: 85%;">
                    {stats_html}
                    <p style="margin: 0; font-size: 14px; color: #e6edf3; white-space: pre-wrap;">{content}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- 10. Input: a photo, a question, or both -----------------------------
# A photo counts as new when it differs from the last one already sent, so
# script re-runs do not keep re-sending the same picture.
new_image = None
if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    fingerprint = (
        uploaded_file.name,
        len(image_bytes),
        hashlib.md5(image_bytes).hexdigest(),
    )
    if fingerprint != st.session_state.get("last_image_fingerprint"):
        new_image = (image_bytes, uploaded_file.type or "image/jpeg", fingerprint)

prompt = st.chat_input("Ask MacroSnap about your meal...")

if prompt or new_image:
    # The chat is created during onboarding; recreate it only if it is missing.
    chat = st.session_state.get("chat")
    if chat is None:
        chat = create_chat()
        st.session_state["chat"] = chat

    # Record the user's message(s) in the visible history.
    if new_image:
        st.session_state["messages"].append(
            {"role": "user", "kind": "image", "content": new_image[0]}
        )
    if prompt:
        st.session_state["messages"].append(
            {"role": "user", "kind": "text", "content": prompt}
        )

    # Build the Gemini request: the photo (if any) plus the question. A bare
    # photo gets the default "what is this meal?" question.
    parts = []
    if new_image:
        parts.append(types.Part.from_bytes(data=new_image[0], mime_type=new_image[1]))
    parts.append(prompt if prompt else IMAGE_ONLY_PROMPT)

    with st.spinner("Crunching the numbers..."):
        try:
            response = chat.send_message(parts)
            answer = response.text
        except Exception as error:
            answer = None
            st.error("Something went wrong while analyzing your meal. Please try again.")

    if answer:
        st.session_state["messages"].append(
            {"role": "assistant", "kind": "text", "content": answer}
        )

        # Update daily totals
        nutrition = parse_nutrition(answer)
        if "daily_totals" not in st.session_state:
            st.session_state["daily_totals"] = {}
        for key, value in nutrition.items():
            num_match = re.search(r"(\d[\d,]*)", value)
            if num_match:
                num = int(num_match.group(1).replace(",", ""))
                st.session_state["daily_totals"][key] = (
                    st.session_state["daily_totals"].get(key, 0) + num
                )

    if new_image:
        st.session_state["last_image_fingerprint"] = new_image[2]


# --- 11. WhatsApp summary card -------------------------------------------
# Only show when there is something to summarize.
has_messages = len(st.session_state.get("messages", [])) > 0
if has_messages:
    st.markdown(
        """
        <div class="card" style="margin-top: 8px;">
            <h3 style="margin-top: 0; font-size: 16px;">Save today's nutrition summary</h3>
            <p style="font-size: 13px; color: #8b949e; margin-bottom: 16px;">Send your MacroSnap summary to WhatsApp.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Send summary to WhatsApp", disabled=not twilio_ready):
        with st.spinner("Summarizing your day..."):
            try:
                response = st.session_state["chat"].send_message(SUMMARY_REQUEST_PROMPT)
                summary = clean_whatsapp_text(response.text)
            except Exception as error:
                summary = None
                st.error("Couldn't generate a summary. Please try again.")

        if summary:
            sent, detail = send_whatsapp(
                st.session_state["whatsapp_number"],
                st.session_state["name"],
                summary,
            )
            if sent:
                st.success("Summary sent to WhatsApp")
            else:
                # Show the ACTUAL Twilio error (code / message / status).
                # Never expose Account SID, Auth Token, or API keys.
                st.error(detail)
                dbg_from = str(st.secrets.get("TWILIO_WHATSAPP_FROM", "")).strip()
                dbg_sid = str(st.secrets.get("TWILIO_CONTENT_SID", "")).strip()
                dbg_raw = str(st.session_state.get("whatsapp_number", "")).strip()
                if dbg_raw.lower().startswith("whatsapp:"):
                    dbg_raw = dbg_raw[len("whatsapp:"):]
                dbg_to = f"whatsapp:{dbg_raw}"
                st.caption(f"From: {dbg_from}")
                st.caption(f"To: {dbg_to}")
                st.caption(f"Content SID: {dbg_sid}")
