from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch

# Load ControlNet models
controlnet = [
    ControlNetModel.from_pretrained("lllyasviel/control_v11f1e_sd15_tile", torch_dtype=torch.float16),
    ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16),
]

# Load Stable Diffusion model
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)

# Set scheduler to DPM++2M
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Optimize memory usage
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

# Prompts
prompt = "a giant standing in a fantasy landscape, best quality"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

generator = torch.Generator(device="cpu").manual_seed(1)


# Set ControlNet reference images
tile_image = load_image(
    "https://i.imgur.com/BZpv0BK.png"
)
canny_image = load_image(
    "https://i.imgur.com/BZpv0BK.png"
)
images = [tile_image, canny_image]

# Predict image
image = pipe(
    prompt,
    images,
    num_inference_steps=20,
    generator=generator,
    negative_prompt=negative_prompt,
    controlnet_conditioning_scale=[1.0, 0.1],
).images[0]

image.save("./multi_controlnet_output.png")