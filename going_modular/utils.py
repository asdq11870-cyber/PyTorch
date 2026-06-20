from pathlib import Path
import torch
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
    model_save_path: The filepath of where the model is saved
  """
  model_path = Path(target_dir)
  model_path.mkdir(parents=True, exist_ok=True)
  assert model_name.endswith((".pt", ".pth")), "model_name should end with '.pt' or '.pth'"
  model_save_path = model_path / model_name
  print(f"Saving model to {model_save_path}")
  torch.save(obj=model.state_dict(),f=model_save_path)
  return model_save_path

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
  loaded_model = model_class(*args,**kwargs)
  loaded_model.load_state_dict(torch.load(f=model_save_path, map_location=device))
  loaded_model.to(device)
  loaded_model.eval()
  return loaded_model