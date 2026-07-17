import os
from pathlib import Path
from tkinter import Image
import zipfile
from matplotlib import pyplot as plt
import requests
import torch
import random
from typing import Tuple, List, Dict
import torchvision
from timeit import default_timer as timer
from PIL import Image



def display_random_images(dataset: torch.utils.data.Dataset,
                          classes: List[str],
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
  if n > 10:
    n = 10
    display_shape = False
    print("Only 10 images for viewing purposes!")

  if seed:
    random.seed(seed)

  random_index = random.sample(range(len(dataset)), k=n)
  plt.figure(figsize=(16,8))
  for i, random_idx in enumerate(random_index):
    targ_image, targ_label = dataset[random_idx][0], dataset[random_idx][1]
    targ_image_adjust = targ_image.permute(1,2,0)
    plt.subplot(1,n,i+1)
    plt.imshow(targ_image_adjust)
    plt.axis(False)
    title = ""
    if classes:
      title = f"class: {classes[targ_label]}"
      if display_shape:
        title = title + f" | shape: {targ_image_adjust.shape}"
    plt.title(title)



def saving_model(model:torch.nn.Module, model_name:str, target_dir:str):
  """
  This helper functions is for saving an exact replica of the model we are training

  This is done by saving it to a target directory that if doesn't exist will be
  created. Making sure the model ends with the appropriate extension. Finally,
  using torch.save to save it

  Args:
    model: The model we are trying to save
    model_name: The name we give to the model when saving
    target_dir: The target directory we want to save the
    model to

  Returns:
    None
  """
  model_path = Path(target_dir)
  if not model_path.exists():
    model_path.mkdir(parents=True, exist_ok=True)
  assert model_name.endswith((".pt", ".pth")), "model_name should end with '.pt' or '.pth'"
  model_save_path = model_path / model_name
  print(f"Saving model to {model_save_path}")
  torch.save(obj=model.state_dict(),f=model_save_path)

def loading_model(model_class,model_save_path, device,*args,**kwargs):

  """
  This helper function is for loading models of any type that were
  previously saved

  This is done by creating a new instance of a class and using
  torch.load to save the model to that instance then moving it
  to the target device and evaluating it

  Args:
    model_class: The class used to create that model
    model_save_path: The filepath where the model is saved
    device: The device the model ran on
    *args, **kwargs: Any additional arguments used by the class

  Returns:
    loadel_model: It returns the model that we loaded
  """
  if model_class is not None:
    loaded_model = model_class(*args,**kwargs)
  loaded_model.load_state_dict(torch.load(f=model_save_path, map_location=device))
  loaded_model.to(device)
  loaded_model.eval()
  return loaded_model

def set_seeds(seed: int=42):
  """Sets random sets for torch operations.

  Args:
      seed (int, optional): Random seed to set. Defaults to 42.
  """
  # Set the seed for general torch operations
  torch.manual_seed(seed)
  # Set the seed for CUDA torch operations (ones that happen on the GPU)
  torch.cuda.manual_seed(seed)

def download_data(source: str, 
                  destination: str,
                  remove_source: bool = True) -> Path:
  """Downloads a zipped dataset from source and unzips to destination.

  Args:
      source (str): A link to a zipped file containing data.
      destination (str): A target directory to unzip data to.
      remove_source (bool): Whether to remove the source after downloading and extracting.
  
  Returns:
      pathlib.Path to downloaded data.
  
  Example usage:
      download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                    destination="pizza_steak_sushi")
  """
  # Setup path to data folder
  data_path = Path("data/")
  data_path.mkdir(parents=True, exist_ok=True)

  zip_file = data_path / "watch_shoe_fragrance.zip"
  extract_path = data_path / destination

  # download only if needed
  if not extract_path.exists():
    print("Downloading data zip")
    request = requests.get(source)

    with open(zip_file, "wb") as f:
        f.write(request.content)

    print("Unzipping")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(data_path)

    if remove_source:
      os.remove(zip_file)
    
    return extract_path
    
from torch.utils.tensorboard import SummaryWriter #type: ignore

def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str=None):
  """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

  log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

  Where timestamp is the current date in YYYY-MM-DD format.

  Args:
      experiment_name (str): Name of experiment.
      model_name (str): Name of model.
      extra (str, optional): Anything extra to add to the directory. Defaults to None.

  Returns:
      torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

  Example usage:
      # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
      writer = create_writer(experiment_name="data_10_percent",
                              model_name="effnetb2",
                              extra="5_epochs")
      # The above is the same as:
      writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
  """
  from datetime import datetime
  import os

  # Get timestamp of current date (all experiments on certain day live in same folder)
  timestamp = datetime.now().strftime("%d-%m-%Y") 


  log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
  if extra: log_dir = os.path.join(log_dir, extra)
      
  print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
  return SummaryWriter(log_dir=log_dir)

def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.
  Args:
  dir_path (str): target directory

  Returns:
  A print out of:
    number of subdiretories in dir_path
    number of images (files) in each subdirectory
    name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
      print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
  """Finds the class folder names in a target directory.
  
  Assumes target directory is in standard image classification format.

  Args:
      directory (str): target directory to load classnames from.

  Returns:
      Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
  
  Example:
      find_classes("food_images/train")
      >>> (["class_1", "class_2"], {"class_1": 0, ...})
  """
  # 1. Get the class names by scanning the target directory
  classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
  
  # 2. Raise an error if class names not found
  if not classes:
      raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
      
  # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
  class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
  return classes, class_to_idx

def plot_transformed_images(image_paths, transform, n=3, seed=42):
  """Plots a series of random images from image_paths.

  Will open n image paths from image_paths, transform them
  with transform and plot them side by side.

  Args:
      image_paths (list): List of target image paths. 
      transform (PyTorch Transforms): Transforms to apply to images.
      n (int, optional): Number of images to plot. Defaults to 3.
      seed (int, optional): Random seed for the random generator. Defaults to 42.
  """
  random.seed(seed)
  random_image_paths = random.sample(image_paths, k=n)
  for image_path in random_image_paths:
    with Image.open(image_path) as f:
      fig, ax = plt.subplots(1, 2)
      ax[0].imshow(f) 
      ax[0].set_title(f"Original \nSize: {f.size}")
      ax[0].axis("off")

      # Transform and plot image
      # Note: permute() will change shape of image to suit matplotlib 
      # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
      transformed_image = transform(f).permute(1, 2, 0) 
      ax[1].imshow(transformed_image) 
      ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
      ax[1].axis("off")

      fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

def model_size_and_params(model, model_savepath) -> Dict:
  model_size = Path(model_savepath).stat().st_size // (1024**2)
  total_params = sum(torch.numel(param) for param in model.parameters())
  return {"MODEL_SIZE":model_size,"TOTAL_PARAMETERS":total_params}


