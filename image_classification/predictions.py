import torchvision
from typing import List, Dict
import torch
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from pathlib import Path
from PIL import Image


def predict(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    show_image:bool = False
):
  """Makes a prediction on a target image with a trained model and plots the image.

  Args:
      model (torch.nn.Module): trained PyTorch image classification model.
      image_path (str): filepath to target image.
      class_names (List[str], optional): different class names for target image. Defaults to None.
      transform (_type_, optional): transform of target image. Defaults to None.
      device (torch.device, optional): target device to compute on. Defaults to "cuda" if torch.cuda.is_available() else "cpu".
      show_image: If true will show the image, if false will return the prediction time and a dictionary of 
      probabilities and labels
  
  Returns:
      Matplotlib plot of target image and model prediction as title.
      The predicted time and a dictionary of labels and probabilities

  Example usage:
      pred_and_plot_image(model=model,
                          image="some_image.jpeg",
                          class_names=["class_1", "class_2", "class_3"],
                          transform=torchvision.transforms.ToTensor(),
                          device=device)
  """
  start = timer()
  if show_image:
      target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

      target_image = target_image / 255.0

  if transform:
      target_image = transform(target_image)

  model.to(device)

  model.eval()
  with torch.inference_mode():
      target_image = target_image.unsqueeze(dim=0)

      target_image_pred = model(target_image.to(device))

  target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

  target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

  target_image_label_and_probs = {class_names[i] : float(target_image_pred_probs[0][i]) for i in range(len(class_names))}

  pred_time = round(timer() - start, 5)

  if show_image:
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)

  if not show_image:
    return target_image_label_and_probs, pred_time
    
def pred_and_store(test_dir, model:torch.nn.Module, transform:torchvision.transforms.Compose, class_names: List[str],
                   device:torch.device) -> List[Dict]:

  """
  This function tells the user the predicted class and how certain the model
  was during it's prediction. It returns a dictionary of the image path, class
  name, predicted probability, predicted class, time taken to make the prediction
  and whether the model was correct

  Args:
    test_dir: The test directory containing all the test images
    model: The model being tested
    transform: The transform applied to the image
    class_names: The names of the classes
    device: The device the model is running on

  Returns:
    A list of dictionaries
  """
  
  test_paths = list(Path(test_dir).glob("*/*.jpg"))
  pred_list = []

  model.to(device)
  model.eval()

  for path in test_paths:
    pred_dict = {}
    pred_dict["image_path"] = path
    class_name = path.parent.stem
    pred_dict["class_name"] = class_name

    start = timer()
    image = Image.open(path).convert("RGB")
    t_image = transform(image).unsqueeze(dim=0).to(device)
    
    with torch.inference_mode():
      pred_logit = model(t_image)
      pred_prob = torch.softmax(pred_logit, dim=1)
      pred_label = torch.argmax(pred_prob, dim=1)
      pred_class = class_names[pred_label.cpu()]
      pred_dict["pred_prob"] = round(pred_prob.max().cpu().item(), 4)
      pred_dict["pred_class"] = pred_class

      end = timer()
      pred_dict["time_for_pred"] = round(end-start, 4)

    pred_dict["correct"] = class_name == pred_class

    pred_list.append(pred_dict)

  return pred_list