from transformers import pipeline
import numpy as np

# Load emotion analysis model (small version for demo)
emotion_classifier = pipeline("text-classification", model="finiteautomata/bertweet-base-emotion-analysis")

def analyze_emotion(text):
    """
    Analyze the emotion in the given text
    Returns a dictionary with emotion labels and scores
    """
    try:
        result = emotion_classifier(text)
        return {
            "dominant_emotion": result[0]['label'],
            "emotion_scores": {r['label']: r['score'] for r in result}
        }
    except:
        return {
            "dominant_emotion": "neutral",
            "emotion_scores": {"neutral": 1.0}
        }