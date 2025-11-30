"""
GenArt Studio - Backend API
FastAPI server for prompt enhancement and image generation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import requests as http_requests
import base64
from io import BytesIO

# Load environment variables
load_dotenv()

app = FastAPI(title="GenArt Studio API")

# Allow requests from the Streamlit frontend (running on port 8501)
origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#  Models 

class PromptEnhanceRequest(BaseModel):
    prompt: str
    style: Optional[str] = None

class GenerateImageRequest(BaseModel):
    prompt: str
    style: Optional[str] = None
    steps: Optional[int] = 30
    cfg_scale: Optional[float] = 7.5

#  Endpoints 

@app.get("/")
async def root():
    return {"message": "GenArt Studio API"}

@app.post("/enhance_prompt")
async def enhance_prompt(request: PromptEnhanceRequest):
    """Enhance a user's prompt using OpenAI GPT-4o mini"""
    try:
        # Create system prompt for art direction
        system_prompt = """You are a professional art director and prompt engineer for AI image generation.
Transform the user's simple description into a detailed, structured artistic prompt suitable for Stable Diffusion.

Guidelines:
- Add specific artistic details (composition, lighting, style, quality)
- Include technical terms (4K, cinematic, high detail, etc.)
- Keep it concise but descriptive (2-3 sentences max)
- Maintain the user's core concept
- Don't add watermarks or text overlays"""

        # Add style context if provided
        user_content = request.prompt
        if request.style and request.style != "None":
            user_content = f"Style: {request.style}\nPrompt: {request.prompt}"
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        enhanced_prompt = response.choices[0].message.content.strip()
        
        return {
            "original_prompt": request.prompt,
            "enhanced_prompt": enhanced_prompt,
            "style": request.style
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prompt enhancement failed: {str(e)}")

@app.post("/generate")
async def generate_image(request: GenerateImageRequest):
    """Generate an image using local Stable Diffusion with diffusers"""
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        
        print(f"Loading Stable Diffusion model (this may take a minute on first run)...")
        
        # Load a smaller, faster model (~1.7GB instead of 4GB)
        model_id = "segmind/small-sd"  # Faster, smaller alternative
        
        # Check if CUDA (GPU) is available, otherwise use CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load pipeline with appropriate dtype
        if device == "cuda":
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                safety_checker=None  # Disable for faster generation
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                safety_checker=None
            )
        
        pipe = pipe.to(device)
        
        print(f"Generating image with prompt: {request.prompt[:50]}...")
        
        # Generate image
        image = pipe(
            prompt=request.prompt,
            num_inference_steps=request.steps or 30,
            guidance_scale=request.cfg_scale or 7.5
        ).images[0]
        
        # Convert PIL image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        print(f"Image generated successfully!")
        return {
            "image": image_base64,
            "prompt": request.prompt,
            "style": request.style
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image generation error: {str(e)}")

@app.get("/styles")
async def get_styles():
    """Load available style presets from JSON file"""
    try:
        styles_path = Path(__file__).parent.parent / "styles" / "presets.json"
        with open(styles_path, 'r') as f:
            data = json.load(f)
            return data.get("styles", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load styles: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
