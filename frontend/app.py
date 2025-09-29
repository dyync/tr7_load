import gradio as gr
import requests
import os
import base64
from PIL import Image
import io
import tempfile

# Model API configuration
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://localhost:1378")

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def base64_to_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

def preprocess_image(image: Image.Image):
    """Send image to model API for preprocessing"""
    try:
        image_data = image_to_base64(image)
        response = requests.post(
            f"{MODEL_API_URL}/preprocess",
            json={"image_data": image_data}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result["success"]:
                processed_image = base64_to_image(result["processed_image"])
                return result["trial_id"], processed_image
        return None, image
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None, image

def image_to_3d(trial_id: str, seed: int, randomize_seed: bool, 
                ss_guidance_strength: float, ss_sampling_steps: int, 
                slat_guidance_strength: float, slat_sampling_steps: int,
                image: Image.Image):
    """Generate 3D model via API"""
    try:
        image_data = image_to_base64(image)
        
        response = requests.post(
            f"{MODEL_API_URL}/generate",
            json={
                "image_data": image_data,
                "seed": seed,
                "randomize_seed": randomize_seed,
                "ss_guidance_strength": ss_guidance_strength,
                "ss_sampling_steps": ss_sampling_steps,
                "slat_guidance_strength": slat_guidance_strength,
                "slat_sampling_steps": slat_sampling_steps
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            if result["success"]:
                # Save video to temporary file
                video_data = result["video"]
                if video_data.startswith('data:video/mp4;base64,'):
                    video_data = video_data.split(',')[1]
                
                video_bytes = base64.b64decode(video_data)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                    f.write(video_bytes)
                    video_path = f.name
                
                return result["state"], video_path
                
        return None, None
    except Exception as e:
        print(f"Generation error: {e}")
        return None, None

def extract_glb(state: dict, mesh_simplify: float, texture_size: int):
    """Extract GLB file via API"""
    try:
        response = requests.post(
            f"{MODEL_API_URL}/extract_glb",
            json={
                "state": state,
                "mesh_simplify": mesh_simplify,
                "texture_size": texture_size
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            if result["success"]:
                glb_data = result["glb_data"]
                filename = result["filename"]
                
                # Return the download path
                return f"{MODEL_API_URL}/download/{filename}"
                
        return None
    except Exception as e:
        print(f"GLB extraction error: {e}")
        return None

def activate_button():
    return gr.Button(interactive=True)

def deactivate_button():
    return gr.Button(interactive=False)

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("""
    ## Image to 3D Asset with [TRELLIS](https://trellis3d.github.io/)
    * Upload an image and click "Generate" to create a 3D asset. 
    * If you find the generated 3D asset satisfactory, click "Extract GLB" to extract the GLB file and download it.
    """)
    
    with gr.Row():
        with gr.Column():
            image_prompt = gr.Image(label="Image Prompt", image_mode="RGBA", type="pil", height=300)
            
            with gr.Accordion(label="Generation Settings", open=False):
                seed = gr.Slider(0, 2147483647, label="Seed", value=0, step=1)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                gr.Markdown("Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                gr.Markdown("Stage 2: Structured Latent Generation")
                with gr.Row():
                    slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1)
                    slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)

            generate_btn = gr.Button("Generate")
            
            with gr.Accordion(label="GLB Extraction Settings", open=False):
                mesh_simplify = gr.Slider(0.9, 0.98, label="Simplify", value=0.95, step=0.01)
                texture_size = gr.Slider(512, 2048, label="Texture Size", value=1024, step=512)
            
            extract_glb_btn = gr.Button("Extract GLB", interactive=False)

        with gr.Column():
            video_output = gr.Video(label="Generated 3D Asset", autoplay=True, loop=True, height=300)
            download_glb = gr.DownloadButton(label="Download GLB", interactive=False)
            
    trial_id = gr.Textbox(visible=False)
    output_buf = gr.State()

    # Handlers
    image_prompt.upload(
        preprocess_image,
        inputs=[image_prompt],
        outputs=[trial_id, image_prompt],
    )
    
    image_prompt.clear(
        lambda: '',
        outputs=[trial_id],
    )

    generate_btn.click(
        image_to_3d,
        inputs=[trial_id, seed, randomize_seed, ss_guidance_strength, ss_sampling_steps, 
                slat_guidance_strength, slat_sampling_steps, image_prompt],
        outputs=[output_buf, video_output],
    ).then(
        activate_button,
        outputs=[extract_glb_btn],
    )

    video_output.clear(
        deactivate_button,
        outputs=[extract_glb_btn],
    )

    extract_glb_btn.click(
        extract_glb,
        inputs=[output_buf, mesh_simplify, texture_size],
        outputs=[download_glb],
    ).then(
        activate_button,
        outputs=[download_glb],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)