import torchvision.models as models

def convnext_tiny():
    return models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)

def convnext_small():
    return models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)

def convnext_base():
    return models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)

def convnext_large():
    return models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)