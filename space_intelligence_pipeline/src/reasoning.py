"""
Reasoning module using Gemini API (grounded).
Takes YOLOv8 crater detection output and produces a plain-English,
honesty-constrained analysis — explicitly flags low-confidence detections.
"""
import json
import os
from google import genai


class CraterReasoner:
    def __init__(self, api_key: str = None, model: str = "gemini-3.6-flash"):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("No Gemini API key provided (arg or GEMINI_API_KEY env var).")
        self.client = genai.Client(api_key=key)
        self.model = model

    def _build_prompt(self, detection_result: dict) -> str:
        detections_json = json.dumps(detection_result["detections"], indent=2)
        return f"""You are analyzing crater detection results from a YOLOv8 model trained on lunar/Martian surface imagery.

Detection summary:
- Total craters detected: {detection_result['count']}
- Raw detections (label, confidence 0-1, box in pixel xyxy):
{detections_json}

Write a plain-English analysis for a non-technical reader. Rules:
1. Only describe what the data actually shows - do not invent details not present in the detections.
2. Explicitly call out any detection with confidence below 0.5 as low-confidence and worth manual review, rather than stating it as a confirmed crater.
3. If detections cluster in one region of the image, mention that spatial pattern.
4. Keep it to 3-5 sentences, no headers or bullet points.
5. If count is 0, say so plainly and suggest possible reasons (image quality, no craters present, model limitations) rather than guessing.
"""

    def reason(self, detection_result: dict) -> str:
        prompt = self._build_prompt(detection_result)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text
