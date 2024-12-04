from transformers import CLIPModel

class clip:
    def __init__(self, name):
        self.model = CLIPModel.from_pretrained(name)
    
    def __call__(self, inputs):
        return self.model.get_image_features(pixel_values=inputs)
    
    def __getattr__(self, name):
        return getattr(self.model, name)

class clip_vitb32(clip):
    def __init__(self):
        super().__init__("openai/clip-vit-base-patch32")

class clip_vitb16(clip):
    def __init__(self):
        super().__init__("openai/clip-vit-base-patch16")

class clip_vitl14(clip):
    def __init__(self):
        super().__init__("openai/clip-vit-large-patch14")