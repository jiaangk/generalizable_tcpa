import torchvision.models as models

def swin_v2_t_transform():
    return models.Swin_V2_T_Weights.DEFAULT.transforms()

def swin_v2_s_transform():
    return models.Swin_V2_S_Weights.DEFAULT.transforms()

def swin_v2_b_transform():
    return models.Swin_V2_B_Weights.DEFAULT.transforms()