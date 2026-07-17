import torch 
from torch.utils.data import Dataset
from PIL import Image
from image_classification.utils import find_classes
from pathlib import Path
from typing import Tuple, Dict, List


class ImageFolderCustom(Dataset):
  def __init__(self, target_dir:str, transform=None) -> None:
    self.paths = list(Path(target_dir).glob("*/*.jpg"))
    self.transforms = transform
    self.classes, self.class_to_idx = find_classes(target_dir)

  def load_image(self, index:int) -> Image.Image:
    image_path = self.paths[index]
    return Image.open(image_path)

  def __len__(self) -> int:
    return len(self.paths)

  def __getitem__(self, index:int) -> Tuple[torch.Tensor,int]:
    image = self.load_image(index)
    class_names = self.paths[index].parent.name
    class_idx = self.class_to_idx[class_names]

    if self.transforms:
      return self.transforms(image), class_idx
    else:
      return image, class_idx