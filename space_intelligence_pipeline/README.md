# Space Image Intelligence Pipeline

> **Ground-to-Orbit Intelligence: AI-Driven Surface Analysis & Live Space Data Q&A**

---

## Module Overview

The **Space Image Intelligence Pipeline** is an end-to-end AI framework engineered to analyze high-resolution extraterrestrial surface imagery (such as Lunar and Martian orbital datasets). By integrating state-of-the-art computer vision models with large language model capabilities, the pipeline transforms raw, noisy planetary imagery into actionable mission intelligence for space exploration and landing site evaluation.

The architecture leverages **Real-ESRGAN** for super-resolution image enhancement, recovering intricate planetary surface topographies and subtle micro-textures. Enhanced frames are fed into a fine-tuned **YOLOv8** object detection model to identify critical geological features including impact craters, boulder fields, dynamic dust shadows, and navigation hazards. Finally, multimodal natural language reasoning powered by the **Gemini API** provides contextualized, physics-aware surface intelligence with rigorous uncertainty and confidence framing.

---

## Core Features List

- **Image Enhancement**: High-fidelity super-resolution restoring fine geological textures using Real-ESRGAN models.
- **Surface Object Detection**: Real-time identification and bounding of lunar/Martian craters, rocks, and surface landing hazards via fine-tuned YOLOv8.
- **Grounded AI Reasoning**: Contextually aware multimodal surface analysis via Gemini API, incorporating strict confidence ratings and domain-specific uncertainty framing.
- **Interactive Grounded Chat**: Conversational interface enabling follow-up Q&A directly grounded on processed satellite and surface visual context.
- **Live Space Data Q&A**: Seamless REST integration with NASA Open APIs (APOD - Astronomy Picture of the Day & NeoWs - Near Earth Object Web Service) for live cosmic updates.
- **Comparative Analysis**: Dual-viewport side-by-side surface change detection and structural comparison across spatial timelines.
- **Mission Report Export**: Automated single-click PDF generation capturing visual overlays, detection metrics, and AI mission assessments.

---

## Tech Stack

- **Core Runtime**: Python 3.10+
- **Deep Learning Framework**: PyTorch
- **Super-Resolution**: Real-ESRGAN
- **Object Detection**: YOLOv8 (`ultralytics`)
- **Multimodal AI Reasoning**: Gemini API (`google-genai` / Google AI Studio)
- **Dashboard & UI**: Streamlit
- **Space Data Integration**: NASA Open APIs (APOD & NeoWs)
- **Document Generation**: ReportLab / FPDF2

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/AI_Analyzer_SIH.git
cd AI_Analyzer_SIH/space_intelligence_pipeline
```

### 2. Environment Setup & Dependencies
It is recommended to use a virtual environment (e.g., `venv` or `conda`):

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1

# Activate on Linux/macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. API Key Configuration
Create a `.env` file in the root of `space_intelligence_pipeline/` or set environment variables:

```bash
# Environment Variables
GEMINI_API_KEY="your_gemini_api_key_here"
NASA_API_KEY="your_nasa_api_key_here"
```

Or copy the provided template:
```bash
cp .env.example .env
```

### 4. Running the Dashboard
Launch the interactive Streamlit mission dashboard:
```bash
streamlit run src/app.py
```
