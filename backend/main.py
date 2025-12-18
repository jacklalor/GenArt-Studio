"""
GenArt Studio - Backend API
FastAPI server for prompt enhancement and image generation
"""

# FastAPI imports for web framework
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # For cross-origin requests

# Pydantic for request/response data validation
from pydantic import BaseModel
from typing import Optional

# Standard library imports
import os  # For environment variables
import json  # For reading style presets
from pathlib import Path  # For file path handling

# Third-party imports
from dotenv import load_dotenv  # Load .env file
from openai import OpenAI  # OpenAI API client
import requests as http_requests  # HTTP requests (if needed)
import base64  # For encoding images as base64 strings
from io import BytesIO  # For handling image data in memory

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI application
app = FastAPI(title="GenArt Studio API")

# Configure CORS (Cross-Origin Resource Sharing)
# This allows the frontend (running on port 8501) to call this API (port 8000)
origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow requests from these origins
    allow_credentials=True,  # Allow cookies/auth headers
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Data Models (Request/Response schemas)

class PromptEnhanceRequest(BaseModel):
    """Request model for prompt enhancement endpoint."""
    prompt: str  # User's original text description
    style: Optional[str] = None  # Optional art style selection

class GenerateImageRequest(BaseModel):
    """Request model for image generation endpoint."""
    prompt: str  # Text description of image to generate
    style: Optional[str] = None  # Optional art style (e.g., "Photorealistic", "Anime")
    model: Optional[str] = "dalle3"  # Generation mode: "demo" or "stable-diffusion"
    steps: Optional[int] = 30  # Number of inference steps (for Stable Diffusion)
    cfg_scale: Optional[float] = 7.5  # Guidance scale (for Stable Diffusion)

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint - returns API status message."""
    return {"message": "GenArt Studio API"}

@app.post("/enhance_prompt")
async def enhance_prompt(request: PromptEnhanceRequest):
    """Enhance a user's prompt using OpenAI GPT-4o mini.
    
    Takes a simple description and transforms it into a detailed artistic prompt
    suitable for image generation models like Stable Diffusion.
    
    Args:
        request: PromptEnhanceRequest containing prompt and optional style
        
    Returns:
        Dict with original_prompt, enhanced_prompt, and style
    """
    try:
        # Create system prompt that instructs GPT-4o on how to enhance prompts
        # This acts as the "personality" and instructions for the AI
        system_prompt = """You are a professional art director and prompt engineer for AI image generation.
Transform the user's simple description into a detailed, structured artistic prompt suitable for Stable Diffusion.

Guidelines:
- Add specific artistic details (composition, lighting, style, quality)
- Include technical terms (4K, cinematic, high detail, etc.)
- Keep it concise but descriptive (2-3 sentences max)
- Maintain the user's core concept
- Don't add watermarks or text overlays"""

        # Prepare user content - add style context if provided
        user_content = request.prompt
        if request.style and request.style != "None":
            # Include style in the prompt for context-aware enhancement
            user_content = f"Style: {request.style}\nPrompt: {request.prompt}"
        
        # Call OpenAI API with chat completions
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use GPT-4o mini (cost-effective, fast)
            messages=[
                {"role": "system", "content": system_prompt},  # AI instructions
                {"role": "user", "content": user_content}  # User's prompt
            ],
            temperature=0.7,  # Balance between creativity (1.0) and consistency (0.0)
            max_tokens=150  # Limit response length to keep prompts concise
        )
        
        # Extract the enhanced prompt from API response
        enhanced_prompt = response.choices[0].message.content.strip()
        
        # Return structured response
        return {
            "original_prompt": request.prompt,
            "enhanced_prompt": enhanced_prompt,
            "style": request.style
        }
        
    except Exception as e:
        # Handle any errors (API failures, network issues, etc.)
        raise HTTPException(status_code=500, detail=f"Prompt enhancement failed: {str(e)}")

@app.post("/generate")
async def generate_image(request: GenerateImageRequest):
    """Generate an image using demo mode or Stable Diffusion.
    
    Routes to appropriate generation function based on selected model:
    - "demo": Fast procedural generation for testing
    - "stable-diffusion": Full AI generation (slower but higher quality)
    
    Args:
        request: GenerateImageRequest with prompt, style, and model selection
        
    Returns:
        Dict containing base64-encoded image, prompt, and style
    """
    try:
        # Route to appropriate generation function based on model selection
        if request.model == "demo":
            return await generate_demo_image(request)
        else:
            return await generate_stable_diffusion_image(request)
    except Exception as e:
        # Handle errors from either generation function
        raise HTTPException(status_code=500, detail=f"Image generation error: {str(e)}")

async def generate_demo_image(request: GenerateImageRequest):
    """Generate a demo image using procedural generation.
    
    Creates a placeholder image using PIL (Python Imaging Library) with:
    - Style-aware color gradients
    - Random shapes for variety
    - Text overlay showing prompt
    
    This mode is instant (<1 second) and uses minimal memory (~50MB).
    Useful for UI testing and demonstrations on low-end hardware.
    
    Args:
        request: GenerateImageRequest with prompt and style
        
    Returns:
        Dict with base64-encoded image, prompt, and style
    """
    from PIL import Image, ImageDraw, ImageFont
    import random
    
    # Extract style name and convert to lowercase for comparison
    style_lower = (request.style or "").lower()
    
    # Set image dimensions
    width, height = 512, 512
    
    # Create blank RGB image
    image = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(image)
    
    # Generate consistent random colors based on prompt hash
    # Same prompt always produces same "random" colors
    seed = hash(request.prompt) % 1000
    random.seed(seed)
    
    # Apply style-specific color schemes
    if "black and white" in style_lower or "monochrome" in style_lower:
        # Generate grayscale gradient from black to white
        for y in range(height):
            gray = int(255 * (y / height))  # 0-255 gradient
            draw.rectangle([(0, y), (width, y + 1)], fill=(gray, gray, gray))
            
    elif "futuristic" in style_lower or "cyberpunk" in style_lower:
        # Generate neon/cyberpunk colors (cyan, magenta, purple)
        for y in range(height):
            r = int(255 * (y / height) * random.uniform(0.3, 0.8))
            g = int(100 * (1 - y / height) * random.uniform(0.5, 1.0))
            b = random.randint(150, 255)  # High blue for neon effect
            draw.rectangle([(0, y), (width, y + 1)], fill=(r, g, b))
            
    else:
        # Default: colorful gradient with random variations
        for y in range(height):
            r = int(255 * (y / height) * random.uniform(0.5, 1.0))
            g = int(255 * (1 - y / height) * random.uniform(0.5, 1.0))
            b = random.randint(100, 200)
            draw.rectangle([(0, y), (width, y + 1)], fill=(r, g, b))
    
    # Add random circular shapes for visual interest
    # Skip for black and white style to maintain minimalism
    if "black and white" not in style_lower:
        for _ in range(5):  # Draw 5 random circles
            x = random.randint(0, width)
            y = random.randint(0, height)
            size = random.randint(30, 100)
            
            # Use brighter colors for futuristic style
            if "futuristic" in style_lower:
                color = (random.randint(0, 255), random.randint(200, 255), random.randint(200, 255))
            else:
                color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            
            draw.ellipse([x, y, x + size, y + size], fill=color, outline=None)
    
    # Add text overlay showing prompt
    try:
        # Try to load Arial font at size 16
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        # Fall back to default font if Arial not available
        font = ImageFont.load_default()
    
    # Create semi-transparent overlay for text background
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))  # Transparent
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Draw dark semi-transparent rectangle at bottom for text
    text_box_height = 60
    overlay_draw.rectangle(
        [(0, height - text_box_height), (width, height)],
        fill=(0, 0, 0, 180)  # Black with 70% opacity
    )
    
    # Merge transparent overlay with main image
    image = image.convert('RGBA')  # Convert to support transparency
    image = Image.alpha_composite(image, overlay)
    image = image.convert('RGB')  # Convert back to RGB for final output
    
    # Draw text on the overlay
    draw = ImageDraw.Draw(image)
    
    # Truncate prompt if too long to fit
    prompt_text = request.prompt if len(request.prompt) <= 70 else request.prompt[:67] + "..."
    
    # Draw app name and prompt
    draw.text((10, height - 50), "GenArt Studio", fill=(255, 255, 255), font=font)
    draw.text((10, height - 25), f"{prompt_text}", fill=(200, 200, 200), font=font)
    
    # Convert PIL image to base64 string for JSON transmission
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # Save to memory as PNG
    image_bytes = buffered.getvalue()  # Get raw bytes
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')  # Encode to base64 string
    
    # Return response matching expected format
    return {
        "image": image_base64,
        "prompt": request.prompt,
        "style": request.style
    }

async def generate_stable_diffusion_image(request: GenerateImageRequest):
    """Generate AI image using Stable Diffusion with memory optimizations.
    
    Uses the segmind/small-sd model (~1.7GB) which is a lightweight variant
    of Stable Diffusion optimized for consumer hardware.
    
    Memory optimizations applied:
    - Reduced resolution (384x384 vs 512x512)
    - Fewer inference steps (15 vs 30)
    - Attention slicing enabled
    
    Generation time: 60-120 seconds on CPU, ~10-20 seconds on GPU
    Memory usage: ~4GB
    
    Args:
        request: GenerateImageRequest with prompt and style
        
    Returns:
        Dict with base64-encoded AI-generated image, prompt, and style
    """
    from diffusers import StableDiffusionPipeline
    import torch
    
    # Apply style modifiers to prompt if style is selected
    prompt = request.prompt
    
    if request.style and request.style != "None":
        # Dictionary mapping style names to keywords that influence SD output
        # These keywords are automatically appended to the user's prompt
        style_modifiers = {
            "Black and White": ", black and white, monochrome, grayscale",
            "Futuristic": ", futuristic, sci-fi, cyberpunk, neon lights, high-tech",
            "Cartoon": ", cartoon style, animated, fun, colorful illustration",
            "Vintage": ", vintage, retro, aged, nostalgic, old photograph",
            "Fantasy": ", fantasy art, magical, mystical, dreamlike",
            "Cinematic": ", cinematic lighting, dramatic, movie quality",
            "Anime": ", anime style, japanese animation",
            "Watercolor": ", watercolor painting, soft colors, artistic",
            "Oil Painting": ", oil painting, classic art style, rich textures",
            "Photorealistic": ", photorealistic, highly detailed, 8k, professional photography",
            "Abstract": ", abstract art, non-representational",
            "Minimalist": ", minimalist, simple, clean design",
            "3D Render": ", 3D render, CGI, computer graphics",
            "Digital Art": ", digital art, vibrant colors, modern",
            "Sketch": ", pencil sketch, hand-drawn, artistic"
        }
        
        # Get modifier for selected style, or create generic one if not in dict
        modifier = style_modifiers.get(request.style, f", {request.style} style")
        
        # Append style keywords to the original prompt
        prompt = prompt + modifier

    # Use lightweight Stable Diffusion model (~1.7GB vs ~4GB for standard)
    model_id = "segmind/small-sd"
    
    # Check if GPU (CUDA) is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Stable Diffusion pipeline from HuggingFace
    if device == "cuda":
        # Use float16 precision on GPU for faster inference and less memory
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # Half precision for GPU
            safety_checker=None  # Disable safety checker for performance
        )
    else:
        # Use full precision on CPU
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            safety_checker=None  # Disable safety checker for performance
        )
    
    # Move pipeline to selected device (GPU or CPU)
    pipe = pipe.to(device)
    
    # Enable attention slicing to reduce memory usage
    # Processes attention in smaller chunks at slight speed cost
    pipe.enable_attention_slicing()
    
    # Generate image with optimized parameters for memory efficiency
    image = pipe(
        prompt=prompt,  # Use style-modified prompt
        num_inference_steps=15,  # Reduced from 30 (faster, less memory, slightly lower quality)
        guidance_scale=7.5,  # How closely to follow the prompt (higher = more literal)
        height=384,  # Reduced from 512 to save memory
        width=384   # Reduced from 512 to save memory
    ).images[0]  # Get first (and only) generated image
    
    # Convert PIL image to base64 string for JSON transmission
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # Save to memory as PNG
    image_bytes = buffered.getvalue()  # Get raw bytes
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')  # Encode to base64
    
    # Return response in expected format
    return {
        "image": image_base64,
        "prompt": request.prompt,  # Return original prompt (not modified version)
        "style": request.style
    }

@app.get("/styles")
async def get_styles():
    """Load and return available art style presets from JSON file.
    
    Reads styles/presets.json which contains a list of style objects,
    each with a 'name' and 'description' field.
    
    Returns:
        List of style objects from the JSON file
    """
    try:
        # Build path to presets.json file
        # Path(__file__).parent = backend/ directory
        # .parent = project root
        # / "styles" / "presets.json" = styles/presets.json
        styles_path = Path(__file__).parent.parent / "styles" / "presets.json"
        
        # Open and parse JSON file
        with open(styles_path, 'r') as f:
            data = json.load(f)
            # Return the "styles" array from the JSON
            return data.get("styles", [])
            
    except Exception as e:
        # Handle file not found, JSON parse errors, etc.
        raise HTTPException(status_code=500, detail=f"Failed to load styles: {str(e)}")

# Entry point when running this file directly (not via uvicorn command)
if __name__ == "__main__":
    import uvicorn
    # Start the FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all network interfaces
        port=8000  # Backend runs on port 8000
    )
