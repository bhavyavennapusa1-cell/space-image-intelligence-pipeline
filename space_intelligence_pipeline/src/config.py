"""
Configuration module for the Space Image Intelligence Pipeline.
Manages environment variables, API endpoints, and model parameter definitions.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")

# REST API Endpoints
NASA_APOD_URL = "https://api.nasa.gov/planetary/apod"
NASA_NEOWS_URL = "https://api.nasa.gov/neo/rest/v1/feed"

# Model Configurations
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
REAL_ESRGAN_MODEL = os.getenv("REAL_ESRGAN_MODEL", "RealESRGAN_x4plus")
