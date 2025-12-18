#  GenArt Studio

**AI-Powered Generative Art Web Application**

A creative web tool that combines Large Language Models (LLMs) with Stable Diffusion to transform simple text prompts into stunning AI-generated artwork.

---

## Overview

GenArt Studio makes AI art creation accessible and fun. Simply describe what you want to see, and the system will:

1. **Enhance your prompt** using an LLM (GPT-4o mini / Llama 3.1) to add artistic detail
2. **Generate artwork** using Stable Diffusion with your chosen style preset
3. **Track your creations and download!** in a visual history panel for easy comparison plus you can choose to save the image

---

## 🚀 Quick Start

### Starting the Application

**Option 1: Using the start script (Windows)**
```powershell
.\start.ps1
```

**Option 2: Manual start (two separate terminals)**

Terminal 1 - Backend:
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Terminal 2 - Frontend:
```bash
python -m streamlit run frontend/app.py
```

Then open your browser to **http://localhost:8501**

---
