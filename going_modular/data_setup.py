import torch # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader  # type: ignore
from torchvision import transforms, datasets # type: ignore
import os
NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):

  """
  Creates training and testing dataloaders

  Takes training and testing directories to create a dataset and then create and
  then create a dataloader and potentially apply a transformation to add variety
  to the data. Other parameters determine the batch size and amount of cpu cores
  working on the training

  Args:
    train_dir: Directory which contains all the training data
    test_dir: Directory which contains all the testing data
    transform: A transformation that can be applied to the dataloader
    batch_size: The amount of data that is funneled into the batch function
    num_workers: The number of cpu cores working on the dataloader

  Returns:
    train_dataloader: The dataloader that funnels the training information
    test_dataloader: The dataloader that funnels the testing information
    class_names: The names of the classes in the dataset
  """

  train_dataset = datasets.ImageFolder(
      root=train_dir, transform=train_transform,

  )
  test_dataset = datasets.ImageFolder(
      root=test_dir, transform=test_transform,

  )

  train_dataloader = DataLoader(
      dataset=train_dataset,
      batch_size=batch_size,
      num_workers=num_workers,
      shuffle=True,
      pin_memory=True
  )

  test_dataloader = DataLoader(
      dataset=test_dataset,
      batch_size=batch_size,
      num_workers=num_workers,
      shuffle=False,
      pin_memory=True
  )

  class_names = train_dataset.classes

  return train_dataloader, test_dataloader, class_names
