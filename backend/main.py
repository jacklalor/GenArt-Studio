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
    model: Optional[str] = "dalle3"  # "dalle3" or "stable-diffusion"
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
    """Generate an image using either demo mode or real Stable Diffusion"""
    try:
        # Check if user selected demo mode
        if request.model == "demo":
            return await generate_demo_image(request)
        else:
            return await generate_stable_diffusion_image(request)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image generation error: {str(e)}")

async def generate_demo_image(request: GenerateImageRequest):
    """Generate a demo image - lightweight placeholder"""
    from PIL import Image, ImageDraw, ImageFont
    import random
    
    print(f"Generating demo image with prompt: {request.prompt[:50]}...")
    
    # Create a colorful gradient image (512x512)
    width, height = 512, 512
    image = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(image)
    
    # Generate random colors based on prompt hash for consistency
    seed = hash(request.prompt) % 1000
    random.seed(seed)
    
    # Create gradient background
    for y in range(height):
        r = int(255 * (y / height) * random.uniform(0.5, 1.0))
        g = int(255 * (1 - y / height) * random.uniform(0.5, 1.0))
        b = random.randint(100, 200)
        draw.rectangle([(0, y), (width, y + 1)], fill=(r, g, b))
    
    # Add some random shapes for variety
    for _ in range(5):
        x = random.randint(0, width)
        y = random.randint(0, height)
        size = random.randint(30, 100)
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        draw.ellipse([x, y, x + size, y + size], fill=color, outline=None)
    
    # Add prompt text overlay
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Add semi-transparent overlay for text
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Text background
    text_box_height = 60
    overlay_draw.rectangle([(0, height - text_box_height), (width, height)], fill=(0, 0, 0, 180))
    
    # Merge overlay
    image = image.convert('RGBA')
    image = Image.alpha_composite(image, overlay)
    image = image.convert('RGB')
    
    # Draw text
    draw = ImageDraw.Draw(image)
    prompt_text = request.prompt if len(request.prompt) <= 70 else request.prompt[:67] + "..."
    draw.text((10, height - 50), "DEMO MODE (instant)", fill=(255, 255, 255), font=font)
    draw.text((10, height - 25), f"{prompt_text}", fill=(200, 200, 200), font=font)
    
    # Convert PIL image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    print(f"Demo image generated!")
    return {
        "image": image_base64,
        "prompt": request.prompt,
        "style": request.style
    }

async def generate_stable_diffusion_image(request: GenerateImageRequest):
    """Generate real AI image using Stable Diffusion (memory-optimized)"""
    from diffusers import StableDiffusionPipeline
    import torch
    
    print(f"Loading Stable Diffusion model...")
    
    # Use smallest available model
    model_id = "segmind/small-sd"  # ~1.7GB
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load pipeline with memory optimizations
    if device == "cuda":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            safety_checker=None
        )
    
    pipe = pipe.to(device)
    
    # Enable memory optimizations
    pipe.enable_attention_slicing()
    
    print(f"Generating AI image (this will take ~60-120 seconds)...")
    
    # Generate smaller image with fewer steps to reduce memory
    image = pipe(
        prompt=request.prompt,
        num_inference_steps=15,  # Reduced from 30 to save memory/time
        guidance_scale=7.5,
        height=384,  # Smaller than default 512
        width=384
    ).images[0]
    
    # Convert PIL image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    print(f"AI image generated successfully!")
    return {
        "image": image_base64,
        "prompt": request.prompt,
        "style": request.style
    }

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
