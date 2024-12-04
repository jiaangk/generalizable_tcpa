import torchvision.models as models

def swin_v2_t():
    return models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT)

def swin_v2_s():
    return models.swin_v2_s(weights=models.Swin_V2_S_Weights.DEFAULT)

def swin_v2_b():
    return models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT)