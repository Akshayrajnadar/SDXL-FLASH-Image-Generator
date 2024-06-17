import gradio as gr
from diffusers import StableDiffusionXLPipeline, DPMSolverSinglestepScheduler
import torch
from PIL import Image
import tempfile
import os
import base64
from huggingface_hub import snapshot_download

# Load the SDXL model
model_path_sdxl = snapshot_download(repo_id="sd-community/sdxl-flash")
pipe_sdxl = StableDiffusionXLPipeline.from_pretrained(
    model_path_sdxl,
    torch_dtype=torch.float16
).to("cuda")

# Configure the scheduler for the SDXL model
pipe_sdxl.scheduler = DPMSolverSinglestepScheduler.from_config(pipe_sdxl.scheduler.config, timestep_spacing="trailing")

def generate_image(prompt, neg_prompt, file_format):
    try:
        # Generate the image
        image = pipe_sdxl(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=7,
            guidance_scale=3
        ).images[0]

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_format}") as tmp_file:
            image_path = tmp_file.name
            image.save(image_path, format=file_format.upper())

        # Read the image as bytes
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()

        # Encode the bytes in base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # Delete the temporary file
        os.remove(image_path)

        # Create a download link
        download_link = f'<a href="data:image/{file_format};base64,{base64_image}" download="generated_image.{file_format}" class="download-button">Download Image</a>'

        return image, download_link
    except Exception as e:
        return None, str(e)

prompt = gr.Text(
    label="Prompt",
    show_label=False,
    max_lines=1,
    placeholder="Enter your prompt",
    container=False,
)
neg_prompt = gr.Text(
    label="Negative Prompt",
    show_label=False,
    max_lines=1,
    placeholder="Enter your negative prompt",
    container=False,
)
file_format = gr.Radio(
    choices=["png", "jpg", "jpeg", "bmp", "tiff", "gif"],
    value="png",
    label="Choose file format"
)

iface = gr.Interface(
    fn=generate_image,
    inputs=[prompt, neg_prompt, file_format],
    outputs=[
        gr.Image(label="Generated Image"),
        gr.HTML(label="Download Link")
    ],
    title="SDXL Model Image Generator",
    allow_flagging=False
)

iface.launch(share=True)
