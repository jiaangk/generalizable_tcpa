import torchvision.models as models

def efficientnet_v2_s_transform():
    return models.EfficientNet_V2_S_Weights.DEFAULT.transforms()

def efficientnet_v2_m_transform():
    return models.EfficientNet_V2_M_Weights.DEFAULT.transforms()

def efficientnet_v2_l_transform():
    return models.EfficientNet_V2_L_Weights.DEFAULT.transforms()