from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import cv2
import base64
from PIL import Image
import numpy as np

def stable_diffusion(prompt):
    model = "runwayml/stable-diffusion-v1-5"
    
    pipeline = StableDiffusionPipeline.from_pretrained(model, torch_dtype = torch.float16)
    pipeline = pipeline.to("cuda")

    image = pipeline(prompt).images[0]  
    
    return image

def stable_diffusion2_1(prompt):
    model = "stabilityai/stable-diffusion-2-1-base"

    scheduler = EulerDiscreteScheduler.from_pretrained(model, subfolder = "scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model, scheduler = scheduler, torch_dtype = torch.float16)
    pipe = pipe.to("cuda")

    negative_prompt = "ugly, hand, mutation"

    image = pipe(prompt, negative_prompt = negative_prompt, num_inference_steps = 200).images[0]

    return image

def encode_image(img, quality = 20):
    _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    encoded = base64.b64encode(buffer)

    return encoded

if __name__ == "__main__":
    img = stable_diffusion2_1("Painting of a stream running through a forest with trees and rocks with vivid color")
    img = img.resize((320, 240), Image.BICUBIC)
    img.save("src/stable_diffusion/stable_diffusion_result.png")
    img = np.array(img)
    
    encoded = encode_image(img, quality = 100)