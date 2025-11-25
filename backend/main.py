"""
GenArt Studio - Backend API
FastAPI server for prompt enhancement and image generation
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="GenArt Studio API")

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
    # Implement LLM integration
    pass

@app.post("/generate")
async def generate_image(request: GenerateImageRequest):
    # Implement Stable Diffusion integration
    pass

@app.get("/styles")
async def get_styles():
    # Load from styles/presets.json
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
