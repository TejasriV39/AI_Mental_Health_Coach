def get_coaching_prompt(user_input, current_emotion, emotion_history):
    """
    Generate a context-aware coaching prompt for Gemini
    """
    # Analyze emotion history trends
    emotion_trend = analyze_emotion_trends(emotion_history)
    
    base_prompt = f"""
    You are an empathetic mental health coach trained in positive psychology and CBT techniques.
    The user is currently feeling: {current_emotion['dominant_emotion']}
    Their emotion trend over recent interactions shows: {emotion_trend}
    
    The user just shared: "{user_input}"
    
    Respond with:
    1. Brief emotional validation (show you understand their feelings)
    2. One thoughtful question to help them explore their feelings
    3. One practical coping strategy tailored to their current emotion
    4. Words of encouragement
    
    Keep your response conversational, compassionate, and under 150 words.
    """
    
    # Add crisis detection and response
    if current_emotion['dominant_emotion'] in ['anger', 'sadness', 'fear'] and current_emotion['emotion_scores'][current_emotion['dominant_emotion']] > 0.9:
        base_prompt += """
        NOTE: The user appears to be in significant distress. Gently suggest professional help options while maintaining support.
        """
    
    return base_prompt

def analyze_emotion_trends(emotion_history):
    """
    Analyze trends in emotion history
    """
    if len(emotion_history) < 3:
        return "not enough data yet"
    
    recent_emotions = [e['dominant_emotion'] for e in emotion_history[-3:]]
    if all(e == recent_emotions[0] for e in recent_emotions):
        return f"consistent {recent_emotions[0]}"
    
    return "mixed emotions"