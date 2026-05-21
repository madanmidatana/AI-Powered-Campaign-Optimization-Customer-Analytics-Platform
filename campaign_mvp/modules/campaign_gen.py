"""
campaign_gen.py — Generate marketing messages via Google Gemini.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Load API key from .env in project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_campaign(segment_name: str, channel: str, top_features: str) -> str:
    """
    Generate a short marketing message for a customer segment using Gemini.

    Parameters
    ----------
    segment_name : str
        E.g. "High Value", "At Risk"
    channel : str
        One of "Email", "SMS", "Push Notification"
    top_features : str
        Comma-separated traits, e.g. "high spend, frequent buyer, recent activity"

    Returns
    -------
    str
        The generated marketing message text.
    """
    prompt = f"""
    Write a short {channel} marketing message for our '{segment_name}' customer segment.
    Key traits: {top_features}.
    Keep it under 60 words. Be personalized and action-oriented.
    Return only the message text, no subject line or labels.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text.strip()
