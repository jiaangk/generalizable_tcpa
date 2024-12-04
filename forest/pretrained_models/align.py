from transformers import AlignModel

class align:
    def __init__(self):
        self.model = AlignModel.from_pretrained('kakaobrain/align-base')
    
    def __call__(self, inputs):
        return self.model.get_image_features(pixel_values=inputs)
    
    def __getattr__(self, name):
        return getattr(self.model, name)
