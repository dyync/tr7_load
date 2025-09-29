from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import imageio
import uuid
from typing import *
from easydict import EasyDict as edict
from PIL import Image
import io
import base64
import json

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils

app = FastAPI(title="Trellis Model API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
print("Loading Trellis pipeline...")
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()
print("Pipeline loaded successfully!")

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = "/tmp/Trellis-demo"

class GenerationRequest(BaseModel):
    image_data: str  # base64 encoded image
    seed: int = 0
    randomize_seed: bool = True
    ss_guidance_strength: float = 7.5
    ss_sampling_steps: int = 12
    slat_guidance_strength: float = 3.0
    slat_sampling_steps: int = 12

class GLBRequest(BaseModel):
    state: dict
    mesh_simplify: float = 0.95
    texture_size: int = 1024

def base64_to_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def pack_state(gs: Gaussian, mesh: MeshExtractResult, trial_id: str) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy().tolist(),
            '_features_dc': gs._features_dc.cpu().numpy().tolist(),
            '_scaling': gs._scaling.cpu().numpy().tolist(),
            '_rotation': gs._rotation.cpu().numpy().tolist(),
            '_opacity': gs._opacity.cpu().numpy().tolist(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy().tolist(),
            'faces': mesh.faces.cpu().numpy().tolist(),
        },
        'trial_id': trial_id,
    }

def unpack_state(state: dict) -> Tuple[Gaussian, edict, str]:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')
    
    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )
    
    return gs, mesh, state['trial_id']

@app.post("/preprocess")
async def preprocess_image(request: GenerationRequest):
    """Preprocess the input image"""
    try:
        image = base64_to_image(request.image_data)
        trial_id = str(uuid.uuid4())
        processed_image = pipeline.preprocess_image(image)
        
        return {
            "trial_id": trial_id,
            "processed_image": image_to_base64(processed_image),
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

@app.post("/generate")
async def generate_3d(request: GenerationRequest):
    """Generate 3D model from image"""
    try:
        image = base64_to_image(request.image_data)
        
        if request.randomize_seed:
            seed = np.random.randint(0, MAX_SEED)
        else:
            seed = request.seed
            
        outputs = pipeline.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=True,
            sparse_structure_sampler_params={
                "steps": request.ss_sampling_steps,
                "cfg_strength": request.ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": request.slat_sampling_steps,
                "cfg_strength": request.slat_guidance_strength,
            },
        )
        
        # Render video
        video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
        video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
        video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
        
        # Save video to bytes
        video_bytes = io.BytesIO()
        imageio.mimsave(video_bytes, video, fps=15, format='mp4')
        video_bytes.seek(0)
        video_base64 = base64.b64encode(video_bytes.read()).decode()
        
        # Pack state
        trial_id = str(uuid.uuid4())
        state = pack_state(outputs['gaussian'][0], outputs['mesh'][0], trial_id)
        
        return {
            "state": state,
            "video": f"data:video/mp4;base64,{video_base64}",
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/extract_glb")
async def extract_glb(request: GLBRequest):
    """Extract GLB file from 3D model state"""
    try:
        gs, mesh, trial_id = unpack_state(request.state)
        glb = postprocessing_utils.to_glb(
            gs, mesh, 
            simplify=request.mesh_simplify, 
            texture_size=request.texture_size, 
            verbose=False
        )
        
        # Export GLB to bytes
        glb_bytes = io.BytesIO()
        glb.export(glb_bytes)
        glb_bytes.seek(0)
        glb_base64 = base64.b64encode(glb_bytes.read()).decode()
        
        return {
            "glb_data": f"data:model/gltf-binary;base64,{glb_base64}",
            "filename": f"{trial_id}.glb",
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GLB extraction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1378)