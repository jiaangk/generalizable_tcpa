import torchvision.models as models

def inception_v3():
    return models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)