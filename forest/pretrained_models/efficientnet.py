import torchvision.models as models

def efficientnet_v2_s():
    return models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

def efficientnet_v2_m():
    return models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)

def efficientnet_v2_l():
    return models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)