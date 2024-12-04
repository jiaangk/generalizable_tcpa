import torchvision.models as models

def resnet18():
    return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

def resnet34():
    return models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

def resnet50():
    return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

def resnet101():
    return models.resnet34(weights=models.ResNet101_Weights.DEFAULT)

def resnet152():
    return models.resnet34(weights=models.ResNet152_Weights.DEFAULT)