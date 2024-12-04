import torchvision.models as models

def inception_v3_transform():
    return models.Inception_V3_Weights.DEFAULT.transforms()