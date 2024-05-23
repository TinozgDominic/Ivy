import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2

class ImageCaptioning():
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype = torch.float16).to("cuda")

    def guess(self, img, condition_text = None):
        if condition_text is not None:
            inputs = self.processor(img, condition_text, return_tensors = "pt").to("cuda", torch.float16)
        else:
            inputs = self.processor(img, return_tensors = "pt").to("cuda", torch.float16)

        out = self.model.generate(**inputs)
        
        return self.processor.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    img_cap = ImageCaptioning()
    img = cv2.imread("/home/tinozg/Documents/Ivy/src/stable_diffusion/stable_diffusion_result.png")
    for _ in range(10):
        txt = img_cap.guess(img)
        print(txt)