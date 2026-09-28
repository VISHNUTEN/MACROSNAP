"""AI Vision Chatbot - a Streamlit app powered by Gemini Vision.

Run it with:  streamlit run app.py
"""

import io
import json
import re

import streamlit as st
from PIL import Image, ImageOps
from google import genai
from google.genai import types
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

from prompts import SYSTEM_PROMPT, DEFAULT_PROMPT

# The Gemini model that can look at images.
# Swap this for another vision model if you want to experiment.
MODEL_NAME = "gemini-2.5-flash"


# --- 1. Build the page ------------------------------------------------
# set_page_config has to be the first Streamlit command in the script.
st.set_page_config(page_title="MACROSNAP 🥗", page_icon="📸")


# --- 2. Onboarding: grab the name and WhatsApp number ---------------
# st.session_state keeps these values for the rest of the browser
# session, so the form only appears once.
if "onboarding_complete" not in st.session_state:
    st.session_state["onboarding_complete"] = False

if not st.session_state["onboarding_complete"]:
    st.title("📸 MACROSNAP")
    st.write("Your instant calorie & macro decoder. Let's get you set up first.")

    with st.form("onboarding_form"):
        name = st.text_input("Your name")

        whatsapp_number = st.text_input(
            "WhatsApp number (with country code)",
            placeholder="+91XXXXXXXXXX",
            help="This is the number MacroSnap will text you on WhatsApp.",
        )

        submitted = st.form_submit_button("Continue")

    # Only validate and save when the form was actually submitted,
    # otherwise the inputs are empty on the first page load.
    if submitted:
        if not name.strip():
            st.error("Please enter your name.")
        elif not whatsapp_number.strip():
            st.error("Please enter your WhatsApp number.")
        else:
            st.session_state["name"] = name.strip()
            st.session_state["whatsapp_number"] = whatsapp_number.strip()
            st.session_state["onboarding_complete"] = True
            # Re-run the script so the app below shows up right away.
            st.rerun()

    # Nothing else to do until the user fills in the form.
    st.stop()


# --- 3. Read the API key from .streamlit/secrets.toml -----------------
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


# --- 4. Read the Twilio credentials from .streamlit/secrets.toml ------
# They stay None when they are not set up yet, so the WhatsApp button can
# explain what to add while the rest of the app keeps working.
# Never print these values.
TWILIO_ACCOUNT_SID = st.secrets.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = st.secrets.get("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = st.secrets.get("TWILIO_WHATSAPP_FROM")

# The Sandbox only delivers template messages to recipients that joined the
# sandbox, so a ContentSid template is used when one is configured. Leaving
# it unset falls back to a plain free-text message, which is all the sandbox
# allows for its own test conversation.
TWILIO_CONTENT_SID = st.secrets.get("TWILIO_CONTENT_SID")

# How many {{1}}, {{2}} ... placeholders the approved template expects.
# Variable 1 gets the user's name, variable 2 gets the MacroSnap summary.
try:
    TWILIO_CONTENT_VAR_COUNT = max(1, int(st.secrets.get("TWILIO_CONTENT_VAR_COUNT", 2)))
except (TypeError, ValueError):
    TWILIO_CONTENT_VAR_COUNT = 2


def read_secret(value):
    """Return a usable secret, treating blanks and 'YOUR_...' as missing.

    This keeps an unfilled placeholder in secrets.toml from being sent to
    Twilio as if it were a real SID or token.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.startswith("YOUR_"):
        return None
    return text


TWILIO_ACCOUNT_SID = read_secret(TWILIO_ACCOUNT_SID)
TWILIO_AUTH_TOKEN = read_secret(TWILIO_AUTH_TOKEN)
TWILIO_WHATSAPP_FROM = read_secret(TWILIO_WHATSAPP_FROM)
TWILIO_CONTENT_SID = read_secret(TWILIO_CONTENT_SID)


def format_whatsapp_number(number):
    """Turn '+91 98765-43210' into 'whatsapp:+919876543210'."""
    raw = str(number).strip().lower()

    # Drop the prefix first, in case the user pasted a full Twilio address.
    if raw.startswith("whatsapp:"):
        raw = raw[len("whatsapp:"):]

    # Keep only the digits and a leading plus, so spaces, dashes and
    # brackets in the pasted number do not break the request.
    cleaned = re.sub(r"[^\d+]", "", raw)
    if not cleaned:
        return ""

    if not cleaned.startswith("+"):
        cleaned = "+" + cleaned

    # E.164 numbers are 8-15 digits. This catches the placeholder text and
    # half-typed numbers here instead of letting Twilio reject them.
    digits = cleaned.lstrip("+")
    if not digits.isdigit() or not 8 <= len(digits) <= 15:
        return ""

    return f"whatsapp:{cleaned}"


def normalize_sender(from_value):
    """Make sure the configured Sandbox sender is a full WhatsApp address.

    Accepts 'whatsapp:+14155238886', '+14155238886' or '14155238886' and
    returns 'whatsapp:+14155238886'. An alphanumeric sender id is left
    untouched.
    """
    raw = str(from_value).strip()
    if not raw:
        return ""

    if raw.lower().startswith("whatsapp:"):
        return raw

    # A numeric sender, with or without a leading plus, needs the channel
    # prefix. Check with a regex, since '+14155238886'.isdigit() is False.
    if re.fullmatch(r"\+?\d+", raw):
        return f"whatsapp:+{raw.lstrip('+')}"

    return raw


def build_content_variables(user_name, summary):
    """Build the JSON string Twilio wants for the template variables.

    The number of keys has to match the placeholders in the template, so it
    is driven by TWILIO_CONTENT_VAR_COUNT.
    """
    values = [str(user_name or "").strip(), str(summary or "").strip()]

    variables = {}
    for index in range(1, TWILIO_CONTENT_VAR_COUNT + 1):
        # Any extra placeholders fall back to the summary rather than
        # sending an empty value.
        variables[str(index)] = values[index - 1] if index <= len(values) else values[-1]

    return json.dumps(variables, ensure_ascii=False)


# The exact text the user has to send from their own WhatsApp app to join the
# Twilio trial. MacroSnap never sends this for them: joining is a WhatsApp
# action that has to happen on the user's own phone.
TWILIO_JOIN_KEYWORD = "join twilio-trial"

# Codes that mean "this recipient is not connected to Twilio yet", so the app
# shows the setup steps again instead of only the raw Twilio message.
TWILIO_SETUP_ERROR_CODES = {57200, 63015, 63016, 21606, 21614, 13268}

# Twilio error codes that deserve a specific, actionable explanation instead of
# a raw API message. The code is matched against TwilioRestException.code.
#
# Twilio reports these in two forms: a 5 digit code (57200) and a 6 digit
# sub-code (572002). Both are the same problem, so only the 5 digit code is
# listed here and twilio_error_hint() also tries the parent code. Trial
# recipients who have not joined the trial are reported as 572002.
TWILIO_ERROR_HINTS = {
    57200: (
        "Your WhatsApp number is not connected to the Twilio Trial yet.\n\n"
        f"Open WhatsApp on the phone you entered during setup and send:\n\n"
        f"{TWILIO_JOIN_KEYWORD}\n\n"
        "to the Twilio WhatsApp number shown in the Twilio Console.\n\n"
        "After Twilio confirms the connection, try Send to WhatsApp again."
    ),
    63015: (
        "Twilio is not allowed to message this number yet. Open WhatsApp on "
        f"the phone you entered during setup and send '{TWILIO_JOIN_KEYWORD}' "
        "to the Twilio WhatsApp number, then try again."
    ),
    63016: (
        "WhatsApp only allows MacroSnap to reply inside an open conversation. "
        f"Message the Twilio WhatsApp number yourself first (for example "
        f"'{TWILIO_JOIN_KEYWORD}'), then click Send to WhatsApp."
    ),
    21606: (
        "This number has not joined the Twilio trial yet. Open WhatsApp on the "
        f"phone you entered during setup and send '{TWILIO_JOIN_KEYWORD}' to "
        "the Twilio WhatsApp number, then try again."
    ),
    21614: (
        "That number is not enabled for WhatsApp. Use the same number you "
        f"sent '{TWILIO_JOIN_KEYWORD}' from, including the country code."
    ),
    13268: (
        "On a Twilio trial the destination has to be a verified recipient. "
        f"Send '{TWILIO_JOIN_KEYWORD}' from that WhatsApp number first, then "
        "try again."
    ),
    21211: (
        "The 'To' number is not a valid WhatsApp address. Enter it with the "
        "country code and no spaces, for example +<country-code><number>."
    ),
}


def twilio_error_hint(code):
    """The friendly explanation for a Twilio error code, or None.

    Twilio returns a 5 digit code such as 57200 and a 6 digit sub-code such
    as 572002 for the same problem. The sub-code is the parent code times
    ten, so when there is no exact match the parent code is tried too.
    """
    hint = TWILIO_ERROR_HINTS.get(code)
    if hint:
        return hint

    parent = code // 10 if isinstance(code, int) and code >= 10000 else None
    return TWILIO_ERROR_HINTS.get(parent)


def is_setup_error(code):
    """True when the code means 'this number has not joined Twilio yet'.

    Checks the sub-code form as well, so 572002 matches 57200.
    """
    if code in TWILIO_SETUP_ERROR_CODES:
        return True

    parent = code // 10 if isinstance(code, int) and code >= 10000 else None
    return parent in TWILIO_SETUP_ERROR_CODES


def verify_content_variables(content_variables, expected_count):
    """Check the built variables match the template's placeholder count.

    Twilio rejects the request when the number of content variables does not
    match the number of {{1}}, {{2}} ... placeholders in the approved
    template, so it is cheaper to catch that here than to debug a 400 from
    the API. Returns an error string, or None when the payload is correct.
    """
    try:
        parsed = json.loads(content_variables)
    except (TypeError, ValueError) as error:
        return f"Could not build the template variables: {error}"

    if not isinstance(parsed, dict):
        return "Template variables must be a JSON object."

    keys = {str(key) for key in parsed}
    expected = {str(index) for index in range(1, expected_count + 1)}

    if keys != expected:
        built = ", ".join(sorted(keys, key=int)) or "none"
        return (
            f"Template variable mismatch: TWILIO_CONTENT_VAR_COUNT is "
            f"{expected_count} but {len(keys)} variable(s) were built "
            f"({built}). Make sure the approved template has that many "
            "{{1}}, {{2}} placeholders."
        )

    return None


def send_whatsapp(to_number, user_name, summary):
    """Send the MacroSnap summary to the user on WhatsApp.

    When TWILIO_CONTENT_SID is configured the message goes out as a
    Content API template: the text comes from the approved template and the
    per-user values travel in content_variables, so no body is sent. Without
    a ContentSid it falls back to a plain free-text message.

    Returns a (sent, detail) pair. On success detail is the message SID,
    on failure it is a message that is safe to show in the UI. The auth
    token is never included in the error text.
    """
    formatted_number = format_whatsapp_number(to_number)
    if not formatted_number:
        return False, "That WhatsApp number does not look right. Use the country code, e.g. +919876543210."

    sender = normalize_sender(TWILIO_WHATSAPP_FROM)
    if not sender:
        return False, "No WhatsApp sender is configured. Set TWILIO_WHATSAPP_FROM in `.streamlit/secrets.toml`."

    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN):
        return False, "Twilio credentials are missing. Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in `.streamlit/secrets.toml`."

    # Build and check the template payload before spending an API call.
    content_variables = None
    if TWILIO_CONTENT_SID:
        content_variables = build_content_variables(user_name, summary)

        mismatch = verify_content_variables(content_variables, TWILIO_CONTENT_VAR_COUNT)
        if mismatch:
            return False, mismatch

    try:
        # The credentials come from .streamlit/secrets.toml, never source.
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        if TWILIO_CONTENT_SID:
            # Template send. Twilio builds the text from the approved
            # template, so the body is driven by the ContentSid and the
            # per-user values go in content_variables.
            message = client.messages.create(
                from_=sender,
                to=formatted_number,
                content_sid=str(TWILIO_CONTENT_SID).strip(),
                content_variables=content_variables,
            )
        else:
            # No template configured, send the summary as free text.
            message = client.messages.create(
                from_=sender,
                to=formatted_number,
                body=summary,
            )

        return True, message.sid

    except TwilioRestException as error:
        # Remember the code so the page can show the setup steps again for
        # 'not connected yet' errors.
        st.session_state["last_twilio_error_code"] = error.code

        # error.code is a short Twilio error code such as 57200, 572002 or
        # 21211. The message is sanitised by Twilio and never contains the
        # token. error.code can be None on transport failures, so match
        # defensively.
        hint = twilio_error_hint(error.code)
        if not hint:
            return False, f"Twilio rejected the message (code {error.code}): {error.msg}"

        return False, f"Twilio rejected the message (code {error.code}):\n\n{hint}"

    except Exception as error:
        # Network hiccups, bad credentials, anything else. Keep the app alive.
        return False, f"Could not send the WhatsApp message: {error}"


def show_twilio_number():
    """The Twilio WhatsApp number the user must message, as plain digits.

    It is read from TWILIO_WHATSAPP_FROM and normalized to '+<number>'.
    Nothing is hardcoded, so the number always matches your Twilio Console.
    """
    return normalize_sender(TWILIO_WHATSAPP_FROM).replace("whatsapp:", "")


def show_whatsapp_trial_setup(open_by_default=False):
    """Explain the Twilio trial steps.

    Joining the trial is something only the user can do, from their own
    WhatsApp app, so this panel explains the steps and shows the exact join
    text. It never sends anything on the user's behalf.
    """
    with st.expander("How to connect WhatsApp", expanded=open_by_default):
        st.markdown("**WhatsApp Trial Setup**")
        st.write(
            "Before sending your first message, connect your WhatsApp number "
            "to the Twilio trial."
        )

        twilio_number = show_twilio_number()
        if twilio_number:
            st.write(f"Send it to the Twilio WhatsApp number `{twilio_number}`.")
        else:
            st.warning(
                "No Twilio number is configured. Add TWILIO_WHATSAPP_FROM to "
                "`.streamlit/secrets.toml` and reload the page."
            )

        st.markdown(
            "1. Open WhatsApp on the phone number you entered here.\n"
            "2. Send this exact message to the Twilio WhatsApp number above:\n"
            "3. Wait for Twilio to confirm you joined, then click "
            "**Send to WhatsApp**."
        )

        # The join keyword Twilio expects, shown so it can be copied exactly.
        st.code(TWILIO_JOIN_KEYWORD, language="text")

        st.caption(
            "You send this message yourself from WhatsApp. MacroSnap cannot "
            "join the trial for you, and it never sends this message for you."
        )


# --- 5. Show the MacroSnap app ----------------------------------------
st.title("📸 MACROSNAP")
st.write("Upload an image, ask a question about it, and Gemini will answer.")


# --- 6. Let the user upload an image and type a question --------------
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

            # Keep the answer in session state, so it survives the re-run
            # that happens when the WhatsApp button is pressed.
            st.session_state["last_image"] = image
            st.session_state["last_summary"] = response.text

        except Exception as error:
            st.error(f"Something went wrong: {error}")


# --- 7. Show the answer, and offer to text it over -------------------
if st.session_state.get("last_image") is not None:
    st.image(st.session_state["last_image"], caption="The image you uploaded")

if st.session_state.get("last_summary"):
    st.markdown("### MACROSNAP's Response")
    st.write(st.session_state["last_summary"])

    twilio_ready = all(
        value and not str(value).startswith("YOUR_")
        for value in (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM)
    )

    if not twilio_ready:
        st.info(
            "WhatsApp is not set up yet. Add TWILIO_ACCOUNT_SID, "
            "TWILIO_AUTH_TOKEN and TWILIO_WHATSAPP_FROM to "
            "`.streamlit/secrets.toml` to enable it."
        )
    elif not TWILIO_CONTENT_SID:
        st.info(
            "No TWILIO_CONTENT_SID is set, so MacroSnap sends a plain "
            "text message. Add the approved template SID as "
            "TWILIO_CONTENT_SID to send through your WhatsApp template."
        )

    # Always point the user at the trial setup, because a trial account
    # cannot send anything until the recipient has joined.
    show_whatsapp_trial_setup(open_by_default=True)

    if st.button(
        "Send to WhatsApp",
        type="primary",
        disabled=not twilio_ready,
    ):
        # The destination is the number captured during onboarding, the same
        # one that worked in the Twilio Console test.
        destination = st.session_state.get("whatsapp_number", "")

        # Clear the previous code so an old error is not reused.
        st.session_state["last_twilio_error_code"] = None

        with st.spinner("Sending your summary on WhatsApp..."):
            sent, detail = send_whatsapp(
                destination,
                st.session_state.get("name", ""),
                st.session_state.get("last_summary", ""),
            )

        if sent:
            st.session_state["last_twilio_error_code"] = None
            st.success("Sent! Check your WhatsApp.")
            st.caption(f"Message SID: {detail}")
        else:
            st.error(detail)

            # Be explicit that this step can only be done from the user's own
            # WhatsApp app, never automatically by MacroSnap.
            if is_setup_error(st.session_state.get("last_twilio_error_code")):
                st.warning(
                    "MacroSnap cannot complete this step for you. It has to be "
                    "done from your own WhatsApp app, using the steps above."
                )
