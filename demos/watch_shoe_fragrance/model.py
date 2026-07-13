import torch
import torchvision
from torch import nn
def create_effnetb2_model(seed:int=32,num_classes:int=3):
  weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
  transform = weights.transforms()
  model = torchvision.models.efficientnet_b2(weights=weights)

  for param in model.features.parameters():
    param.requires_grad = False

  for param in model.features[-2].parameters():
    param.requires_grad = True

  torch.manual_seed(seed)
  model.classifier = nn.Sequential(
      nn.Dropout(p=0.3,inplace=True),
      nn.Linear(in_features=model.classifier[1].in_features,out_features=num_classes)
  )

  return model, transform
