import torch
from transformers import AutoProcessor

class align_transform:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained('kakaobrain/align-base')
    
    def __call__(self, img):
        inputs = self.processor(images=img, return_tensors="pt", do_rescale=False if isinstance(img, torch.Tensor) else True)
        return inputs['pixel_values'].squeeze(0)