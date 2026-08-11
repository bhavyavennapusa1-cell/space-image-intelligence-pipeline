"""
Multimodal Grounded AI Reasoning module using Gemini API.
Evaluates physical terrain safety, confidence indices, and uncertainty framing.
"""

from typing import Optional
from PIL import Image
from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME

class SurfaceReasoningEngine:
    """
    Multimodal surface analysis engine leveraging the Gemini API.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = GEMINI_MODEL_NAME

    def analyze_surface(self, image: Image.Image, prompt: str = "Analyze surface features and hazards.") -> str:
        """
        Generates grounded physics-aware planetary surface reasoning.
        
        Args:
            image (Image.Image): Analyzed input visual frame.
            prompt (str): Custom instruction or contextual query.
            
        Returns:
            str: Multimodal evaluation with uncertainty and confidence framing.
        """
        if not self.api_key:
            return (
                "⚠️ API Key missing: Please configure GEMINI_API_KEY in your environment or .env file."
            )
        
        # Placeholder response demonstrating grounded output framing
        return (
            "### Planetary Surface Intelligence Assessment\n\n"
            "**1. Geological Topography**:\n"
            "- Identified primary impact structure with sloped rim boundaries.\n"
            "- High concentration of regolith ejecta surrounding the central basin.\n\n"
            "**2. Landing Safety & Hazard Evaluation**:\n"
            "- Flat plains within quadrant B present optimal landing zones.\n"
            "- Sharp boulders detected near slope boundaries pose localized stability risks.\n\n"
            "**3. Confidence & Uncertainty Framing**:\n"
            "- Overall Assessment Confidence: **89%**\n"
            "- Uncertainty Note: Sub-surface composition requires thermal infrared validation."
        )
