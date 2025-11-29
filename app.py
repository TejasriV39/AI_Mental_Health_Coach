import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os

# -------------------- Load .env and API Key --------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Add it to your .env or environment variables.")
    st.stop()

# -------------------- GenAI SDK (correct usage) --------------------
import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)

# -------------------- App config --------------------
st.set_page_config(page_title="Emotion-Aware AI Mental Health Coach", page_icon="🤖")
st.title("🤖 Emotion-Aware AI Mental Health Coach")

# -------------------- Load RoBERTa for emotion detection --------------------
@st.cache_resource
def load_roberta_model():
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

roberta_tokenizer, roberta_model = load_roberta_model()

# -------------------- Crisis resources --------------------
CRISIS_RESOURCES = """
**Immediate Help Available:**
- National Suicide Prevention Lifeline: 988 (US)
- Crisis Text Line: Text HOME to 741741 (US)
- International Association for Suicide Prevention: [Find local hotlines](https://findahelpline.com), worldwide.
"""

# -------------------- Session state --------------------
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_first_message" not in st.session_state:
    st.session_state.awaiting_first_message = False
if "mood_history" not in st.session_state:
    st.session_state.mood_history = []
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "streak" not in st.session_state:
    st.session_state.streak = 0
if "last_session_date" not in st.session_state:
    st.session_state.last_session_date = None

# -------------------- Helpers --------------------
def detect_emotion(prompt: str) -> str:
    """Detect primary emotion using RoBERTa"""
    try:
        inputs = roberta_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = roberta_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=1).item()
        emotion_labels = roberta_model.config.id2label
        return emotion_labels[predicted_class].capitalize()
    except Exception as e:
        st.warning(f"Emotion detection failed: {e}")
        return "Neutral"

def generate_exercise(emotion: str) -> str:
    exercises = {
        "Joy": "🌞 Reflect on what made you happy today.",
        "Sadness": "📝 Try writing down 3 things you’re grateful for.",
        "Anger": "❄️ Try the 5-4-3-2-1 grounding technique.",
        "Fear": "🌬️ Deep breathing — inhale 4s, hold 7s, exhale 8s.",
        "Disgust": "🧘 Practice mindfulness meditation for calm focus.",
        "Surprise": "🎯 Take a mindful pause before reacting.",
        "Neutral": "🧠 Practice self-awareness journaling."
    }
    return exercises.get(emotion, "🧘 Mindfulness meditation")

def generate_conversation_title(prompt: str) -> str:
    timestamp = datetime.now().strftime("%b %d")
    return f"Session ({timestamp}) - {prompt[:20]}..."

def _extract_generated_text(response) -> str:
    """Robust extraction for different SDK response shapes."""
    if response is None:
        return None
    # Preferred: response.text
    if hasattr(response, "text") and response.text:
        return response.text
    # Try candidates path
    try:
        # Some SDK versions return nested structure
        cand = getattr(response, "candidates", None)
        if cand and len(cand) > 0:
            # try different nested possibilities:
            first = cand[0]
            # direct text
            if hasattr(first, "output_text"):
                return first.output_text
            # nested output -> content -> text
            out = getattr(first, "output", None)
            if out and len(out) > 0:
                content = getattr(out[0], "content", None)
                if content and len(content) > 0:
                    text = getattr(content[0], "text", None)
                    if text:
                        return text
            # maybe candidate has 'content' list
            if hasattr(first, "content") and len(first.content) > 0:
                maybe = getattr(first.content[0], "text", None)
                if maybe:
                    return maybe
    except Exception:
        pass
    # fallback to str(response)
    try:
        return str(response)
    except Exception:
        return None

def generate_gemini_response(prompt: str, emotion: str) -> str:
    """Use Gemini 1.5 Flash to generate an empathetic coaching response."""
    system_prompt = f"""
You are a compassionate, empathetic mental health coach. Be concise, kind and practical.

The user is feeling: {emotion}

User message:
\"\"\"{prompt}\"\"\"

Respond with:
1) One short validation sentence
2) Three concrete coping strategies (bulleted)
3) One CBT-style cognitive reframe (short)
4) When to seek professional help (short)
End with a simple immediate grounding exercise suggestion.
"""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(system_prompt)
        text = _extract_generated_text(response)
        if not text:
            return "⚠️ Gemini returned an unexpected response format."
        return text
    except Exception as e:
        # helpful error message
        return f"⚠️ Sorry, I couldn’t generate a response: {e}"

# -------------------- Sidebar UI --------------------
with st.sidebar:
    st.header("Mental Health Tools")
    if not st.session_state.user_name:
        name = st.text_input("What's your name?")
        if name:
            st.session_state.user_name = name
            st.success(f"Welcome, {name}! 🌻")
    else:
        st.markdown(f"### 👋 Hi, {st.session_state.user_name}!")

    st.divider()
    st.subheader("Mood Tracker")
    if st.session_state.mood_history:
        mood_df = pd.DataFrame({
            "Date": [datetime.now().strftime("%m/%d")] * len(st.session_state.mood_history),
            "Mood": st.session_state.mood_history
        })
        st.line_chart(mood_df.set_index("Date"))

    st.divider()
    st.subheader("Activity Streak")
    st.markdown(f"🔥 {st.session_state.streak} day streak")

    st.divider()
    st.subheader("Emergency Help")
    st.markdown(CRISIS_RESOURCES)

    st.divider()
    st.subheader("Conversation History")
    if st.button("+ New Session"):
        st.session_state.current_conversation = None
        st.session_state.messages = []
        st.session_state.awaiting_first_message = True
        st.experimental_rerun()

    for title in reversed(list(st.session_state.conversations.keys())):
        if st.button(title, key=title):
            st.session_state.current_conversation = title
            st.session_state.messages = st.session_state.conversations[title]
            st.session_state.awaiting_first_message = False
            st.experimental_rerun()

# -------------------- Main chat area --------------------
if st.session_state.awaiting_first_message:
    st.info("Please share how you're feeling to start your session...")

if prompt := st.chat_input("Share your thoughts or feelings..."):
    today = datetime.now().date()
    if st.session_state.last_session_date != today:
        st.session_state.streak += 1
        st.session_state.last_session_date = today

    emotion = detect_emotion(prompt)
    st.session_state.mood_history.append(emotion)

    if not st.session_state.current_conversation or st.session_state.awaiting_first_message:
        title = generate_conversation_title(prompt)
        st.session_state.current_conversation = title
        st.session_state.conversations[title] = []
        st.session_state.messages = []
        st.session_state.awaiting_first_message = False

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(prompt)
        with col2:
            st.markdown(f"`{emotion}`")

    with st.chat_message("assistant"):
        with st.spinner("Thinking compassionately..."):
            crisis_keywords = ["kill myself", "end my life", "suicide", "self-harm"]
            if any(k in prompt.lower() for k in crisis_keywords):
                reply = f"""
I hear you're in deep pain right now. Please know you’re not alone.

**Immediate Help:**
{CRISIS_RESOURCES}
Would you like help finding professional support near you?
"""
            else:
                reply = generate_gemini_response(prompt, emotion)
                reply += f"\n\n**Try This Now:** {generate_exercise(emotion)}"

            if st.session_state.user_name:
                reply = reply.replace("User", st.session_state.user_name)

        st.markdown(reply)

        if st.session_state.streak % 5 == 0 and st.session_state.streak > 0:
            st.balloons()
            st.success(f"🌟 Great job, {st.session_state.user_name}! You’re on a {st.session_state.streak}-day self-care streak!")

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.conversations[st.session_state.current_conversation] = st.session_state.messages
    st.experimental_rerun()

# -------------------- Display conversation --------------------
if st.session_state.current_conversation and not st.session_state.awaiting_first_message:
    st.subheader(st.session_state.current_conversation)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
elif st.session_state.conversations and not st.session_state.awaiting_first_message:
    most_recent = list(st.session_state.conversations.keys())[-1]
    st.session_state.current_conversation = most_recent
    st.session_state.messages = st.session_state.conversations[most_recent]
    st.experimental_rerun()

# -------------------- Empty state --------------------
if not st.session_state.conversations and not st.session_state.awaiting_first_message:
    st.info("Share how you're feeling to begin your mental health journey 💬")
