import torchvision.models as models

def resnet18_transform():
    return models.ResNet18_Weights.DEFAULT.transforms()

def resnet34_transform():
    return models.ResNet34_Weights.DEFAULT.transforms()

def resnet50_transform():
    return models.ResNet50_Weights.DEFAULT.transforms()

def resnet101_transform():
    return models.ResNet101_Weights.DEFAULT.transforms()

def resnet152_transform():
    return models.ResNet152_Weights.DEFAULT.transforms()