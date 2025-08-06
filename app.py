import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import time

# Initialize Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Use Gemini 1.5 Flash
model = genai.GenerativeModel("gemini-1.5-flash")

# Enhanced System Prompts
SYSTEM_PROMPT = """You are an expert mental health coach with psychology training. Follow these rules:
1. Detect ALL emotions in the user's message and validate them
2. Provide structured coaching responses with:
   - Emotional Support: Validate feelings with empathy
   - Coping Strategies: 3-5 specific, actionable techniques
   - Cognitive Reframing: Help challenge negative thoughts (CBT approach)
   - When to Seek Help: Clear professional referral guidance
3. For high-risk statements (self-harm, suicide), immediately:
   - Acknowledge the seriousness
   - Provide crisis resources
   - Encourage professional help
4. Maintain compassionate but professional tone
5. Incorporate mindfulness and grounding techniques when appropriate"""

TITLE_PROMPT = """Analyze this mental health concern and generate a very concise title (3-5 words max) 
that summarizes the main emotional state. Examples:
- "Anxiety Support"
- "Depression Help"
- "Stress Management"
- "Self-Esteem Boost"

Respond ONLY with the title, nothing else.

User's concern: """

# Crisis resources
CRISIS_RESOURCES = """
**Immediate Help Available:**
- National Suicide Prevention Lifeline: 988 (US)
- Crisis Text Line: Text HOME to 741741 (US)
- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/
"""

# Streamlit app configuration
st.set_page_config(page_title="Emotion-Aware Mental Health Coach", page_icon="🧠")
st.title("🧠 Emotion-Aware AI Mental Health Coach")

# Initialize session state
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

# Emotion detection function
def detect_emotion(prompt):
    """Detect primary emotion from user input"""
    try:
        emotion_prompt = f"""Analyze this message and identify the primary emotion from these options: 
        Happiness, Sadness, Anger, Fear, Anxiety, Stress, Calm, Neutral. 
        Return ONLY the emotion word, nothing else.
        
        Message: "{prompt}" """
        
        response = model.generate_content(
            emotion_prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 10
            }
        )
        return response.text.strip()
    except:
        return "Neutral"

# Generate guided exercise
def generate_exercise(emotion):
    """Generate a guided exercise based on detected emotion"""
    exercises = {
        "Anxiety": "🌬️ 4-7-8 Breathing Exercise",
        "Stress": "🧘‍♂️ Progressive Muscle Relaxation",
        "Sadness": "📝 Gratitude Journaling",
        "Anger": "❄️ Cool Down Countdown",
        "Fear": "🌍 Grounding Technique (5-4-3-2-1)"
    }
    return exercises.get(emotion, "🧠 Mindfulness Meditation")

# Generate conversation title
def generate_conversation_title(prompt):
    """Generate a descriptive title based on user's first message"""
    try:
        response = model.generate_content(
            TITLE_PROMPT + prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 15
            }
        )
        title = response.text.strip('"\'').split('\n')[0]
        timestamp = datetime.now().strftime("%b %d")
        return f"{title} ({timestamp})"
    except:
        timestamp = datetime.now().strftime("%b %d")
        return f"Mental Health Session ({timestamp})"

# Sidebar for features
with st.sidebar:
    st.header("Mental Health Tools")
    
    # User profile
    if not st.session_state.user_name:
        name = st.text_input("What's your name?")
        if name:
            st.session_state.user_name = name
            st.success(f"Welcome, {name}!")
    else:
        st.markdown(f"### 👋 Hi, {st.session_state.user_name}!")
    
    # Mood tracker
    st.divider()
    st.subheader("Mood Tracker")
    if st.session_state.mood_history:
        mood_df = pd.DataFrame({
            "Date": [datetime.now().strftime("%m/%d") for _ in st.session_state.mood_history],
            "Mood": st.session_state.mood_history
        })
        st.line_chart(mood_df.set_index("Date"))
    
    # Streak counter
    st.divider()
    st.subheader("Activity Streak")
    st.markdown(f"🔥 {st.session_state.streak} day streak")
    
    # Crisis resources (always visible)
    st.divider()
    st.subheader("Emergency Help")
    st.markdown(CRISIS_RESOURCES)
    
    # Conversation history
    st.divider()
    st.subheader("Conversation History")
    if st.button("+ New Session"):
        st.session_state.current_conversation = None
        st.session_state.messages = []
        st.session_state.awaiting_first_message = True
        st.rerun()
    
    for title in reversed(list(st.session_state.conversations.keys())):
        if st.button(title, key=title):
            st.session_state.current_conversation = title
            st.session_state.messages = st.session_state.conversations[title]
            st.session_state.awaiting_first_message = False
            st.rerun()

# Main chat interface
if st.session_state.awaiting_first_message:
    st.info("Please share how you're feeling to start your session...")

if prompt := st.chat_input("Share your thoughts or feelings..."):
    # Update streak counter if it's a new day
    today = datetime.now().date()
    if st.session_state.last_session_date != today:
        st.session_state.streak += 1
        st.session_state.last_session_date = today
    
    # Detect emotion and add to history
    emotion = detect_emotion(prompt)
    st.session_state.mood_history.append(emotion)
    
    # Create new conversation if needed
    if not st.session_state.current_conversation or st.session_state.awaiting_first_message:
        title = generate_conversation_title(prompt)
        st.session_state.current_conversation = title
        st.session_state.conversations[title] = []
        st.session_state.messages = []
        st.session_state.awaiting_first_message = False
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message with emotion badge
    with st.chat_message("user"):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(prompt)
        with col2:
            st.markdown(f"`{emotion}`")
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Processing with care..."):
            try:
                # Check for crisis keywords
                crisis_keywords = ["kill myself", "end my life", "suicide", "self-harm"]
                if any(keyword in prompt.lower() for keyword in crisis_keywords):
                    reply = f"""I hear you're in tremendous pain right now. Please know you're not alone.

**Immediate Help:**
{CRISIS_RESOURCES}

Would you like me to help you connect with a professional right now?"""
                else:
                    full_prompt = f"""
                    [User Message]: "{prompt}"
                    [Detected Emotion]: {emotion}
                    
                    As a mental health coach, provide:
                    1. Emotional validation (acknowledge their feelings)
                    2. 3 coping strategies tailored to {emotion}
                    3. One cognitive reframing exercise (CBT approach)
                    4. When to consider professional help"""
                    
                    response = model.generate_content(
                        SYSTEM_PROMPT + full_prompt,
                        generation_config={
                            "temperature": 0.3,
                            "max_output_tokens": 600
                        }
                    )
                    reply = response.text
                    
                    # Add guided exercise
                    exercise = generate_exercise(emotion)
                    reply += f"\n\n**Try This Now:** {exercise}"
                    
                    if st.session_state.user_name:
                        reply = reply.replace("User", st.session_state.user_name)
                    
            except Exception as e:
                reply = "I'm having trouble responding. For immediate support, please contact a mental health professional."
        
        st.markdown(reply)
        
        # Celebration for streaks
        if st.session_state.streak % 5 == 0 and st.session_state.streak > 0:
            st.balloons()
            st.success(f"Amazing! You're on a {st.session_state.streak}-day streak of self-care!")
    
    # Save conversation
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.conversations[st.session_state.current_conversation] = st.session_state.messages
    st.rerun()

# Display current conversation
if st.session_state.current_conversation and not st.session_state.awaiting_first_message:
    st.subheader(st.session_state.current_conversation)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
elif st.session_state.conversations and not st.session_state.awaiting_first_message:
    most_recent = list(st.session_state.conversations.keys())[-1]
    st.session_state.current_conversation = most_recent
    st.session_state.messages = st.session_state.conversations[most_recent]
    st.rerun()

# Empty state
if not st.session_state.conversations and not st.session_state.awaiting_first_message:
    st.info("Share how you're feeling to begin your mental health journey")