import torch
from transformers import AutoProcessor

class clip_transform:
    def __init__(self, name):
        self.processor = AutoProcessor.from_pretrained(name)
    
    def __call__(self, img):
        inputs = self.processor(images=img, return_tensors="pt", do_rescale=False if isinstance(img, torch.Tensor) else True)
        return inputs['pixel_values'].squeeze(0)

class clip_vitb32_transform(clip_transform):
    def __init__(self):
        super().__init__("openai/clip-vit-base-patch32")

class clip_vitb16_transform(clip_transform):
    def __init__(self):
        super().__init__("openai/clip-vit-base-patch16")

class clip_vitl14_transform(clip_transform):
    def __init__(self):
        super().__init__("openai/clip-vit-large-patch14")