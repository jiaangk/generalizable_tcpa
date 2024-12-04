import torchvision.models as models

def convnext_tiny_transform():
    return models.ConvNeXt_Tiny_Weights.DEFAULT.transforms()

def convnext_small_transform():
    return models.ConvNeXt_Small_Weights.DEFAULT.transforms()

def convnext_base_transform():
    return models.ConvNeXt_Base_Weights.DEFAULT.transforms()

def convnext_large_transform():
    return models.ConvNeXt_Large_Weights.DEFAULT.transforms()