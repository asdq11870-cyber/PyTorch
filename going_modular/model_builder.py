import torch #type: ignore
from torch import nn #type: ignore

class TinyVGG(nn.Module):
  """
  Creates a TinyVGG architecture
  Replicates the architecture of the CNN Explainer website
  Args:
    input: The shape of the input size
    hidden: The amount of hidden units there are
    output: The shape of the output size
  """
  def __init__(self,input:int, hidden:int, output:int):
    super().__init__()
    self.relu = nn.ReLU()
    self.conv_layer_1 = nn.Sequential(
        nn.Conv2d(in_channels=input, out_channels=hidden, kernel_size=3, stride=1, padding=1),
        self.relu,
        nn.Conv2d(in_channels=input, out_channels=hidden, kernel_size=3, stride=1, padding=1),
        self.relu,
        nn.MaxPool2d(kernel_size=3, stride=2)
    )
    self.conv_layer_2 = nn.Sequential(
        nn.Conv2d(in_channels=input, out_channels=hidden, kernel_size=3, stride=1, padding=1),
        self.relu,
        nn.Conv2d(in_channels=input, out_channels=hidden, kernel_size=3, stride=1, padding=1),
        self.relu,
        nn.MaxPool2d(kernel_size=3, stride=2)
    )
    self.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(output_size=1),
        nn.Flatten(),
        nn.Linear(hidden, output)
    )

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    return self.classifer(self.conv_layer_2(self.conv_layer_1(x)))