import torch # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter # pyright: ignore[reportMissingImports]

def batch_train(model:torch.nn.Module,
                train_data_loader:torch.utils.data.DataLoader,
                val_data_loader:torch.utils.data.DataLoader,
                test_data_loader:torch.utils.data.DataLoader,
                epochs:int,
                divisor:int,
                device:torch.device,
                loss_curves:bool,
                optimiser:torch.optim,
                loss_fn:torch.nn,
                writer: torch.utils.tensorboard.SummaryWriter | None,
                schedular: torch.optim.lr_schedular.LRSchedular | None):
  """
  Function for training batches of data using dataloaders

  Each batch is iterative given to the function to be experimented
  on and then valed on.

  Args:
    model: The model being valed
    train_data_loader: The dataloader that contains the training data
    val_data_loader: The dataloader that contains the validating data
    test_data_loader: The dataloader that contains the testing data
    epochs: The amount of times the data is trained for
    divisor: Determines how much information is shown
    loss_curves: Determines if at the end loss curves are shown or not
    optimiser: Used for gradient descent
    loss_fn: Used for backpropagation and calculating loss
    writer: This tensorboard summary writer is for writing files and uploading
    them to tensorboard

  Returns:
    Nothing
  """
  start = timer()
  patience = 20 # Increasing from 10 to 20 as the schedular will decrease the lr overtime
  overfit_counter = 0
  epochs_no_imp = 0
  best_loss = float("inf")

  results = {
      "train_loss": [],
      "train_acc": [],
      "val_loss": [],
      "val_acc": []
  }

  if writer is not None:
   writer.add_graph(
     model=model,
     input_to_model=torch.randn(32,3,224,224).to(device)
   )

  for epoch in range(epochs):
      print(f"Epoch: {epoch+1} \n ---------------------------------------------------------")
      train_loss, train_correct, train_total = 0.0, 0, 0
      for batch, (x,y) in enumerate(train_data_loader):
          x,y = x.to(device), y.to(device)
          model.train()
          y_logits = model(x)
          loss = loss_fn(y_logits, y)
          train_loss += loss.item()
          y_pred_labels = y_logits.argmax(dim=1)
          train_correct += (y_pred_labels == y).sum().item()
          train_total += y.size(0)
          optimiser.zero_grad()
          loss.backward()
          optimiser.step()

          #if batch % 400 == 0:
              #print(f"Looked at {batch*len(x)}/{len(data_loader.dataset)} samples")

      train_loss /= len(train_data_loader)
      train_acc = 100 * (train_correct / train_total)

      val_loss, val_correct, val_total = 0.0,0,0
      model.eval()
      with torch.inference_mode():
          for x,y in val_data_loader:
              x,y = x.to(device), y.to(device)
              val_logits = model(x)
              val_loss += loss_fn(val_logits, y).item()
              val_pred_labels = val_logits.argmax(dim=1)
              val_correct += (val_pred_labels == y).sum().item()
              val_total += y.size(0)

          val_loss /= len(val_data_loader)
          val_acc = 100 * (val_correct / val_total)

      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["val_loss"].append(val_loss)
      results["val_acc"].append(val_acc)

      if epoch % divisor == 0:
          print(f"\nTrain Loss: {train_loss:.5f} | Train Accuracy: {train_acc:.2f}% | Validation Loss: {val_loss:.5f} | Validation Accuracy: {val_acc:.2f}%\n")
          overfit_counter = detect_overfitting(results,epoch,overfit_counter)

      best_loss, epochs_no_imp = stagnation(results,epoch,best_loss, epochs_no_imp)
      if(epochs_no_imp >= patience):
          print("Loss is stagnant. Prematurely ending training!")
          break
      
      if schedular is not None:
          schedular.step()

      if writer is not None:
       writer.add_scalars(
           main_tag="Loss",
           tag_scalar_dict={
               "train_loss":train_loss,
               "val_loss":val_loss
           },
           global_step=epoch
       )

       writer.add_scalars(
           main_tag="Accuracy",
           tag_scalar_dict={
               "train_acc":train_acc,
               "val_acc":val_acc
           },
           global_step=epoch
       )
  


  model.eval()
  test_loss, test_correct, test_total = 0,0,0
  with torch.inference_mode():
       for x,y in test_data_loader:
            x,y = x.to(device),y.to(device)
            test_logits=model(x)
            test_loss += loss_fn(test_logits, y).item()
            test_pred = test_logits.argmax(dim=1)
            test_correct += (test_pred == y).sum().item()
            test_total += y.size(0)

  test_loss /= len(test_data_loader)
  test_acc = 100 * (test_correct / test_total)
  print(f"Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}%")

      
      
  if writer is not None: writer.close()
  end = timer()
  print(f"Total Training Time: {end-start:.2f} seconds")

  if loss_curves:
      plot_loss_curves(results)


def detect_overfitting(results,epoch:int,overfit_counter:int):

  """
  This helper function is used for detecting overfitting within the
  model that is being trained. Should the train and val loss become
  to different to each other then an overfitting warning is given

  Args:
    results: A dictionary contains keys of "train_loss", "train_acc",
    "val_loss" and "val_acc" each value is a list appended each epoch
    epoch: The amount of times the data is trained for
    overfit_counter: The amount of times the program has detected overfitting.

  Returns:
    overfit_counter: To be used by the program to see if the model
    is overfitting
  """


  if results["val_loss"][epoch] - results["train_loss"][epoch] > 0.1:
      overfit_counter += 1
  else:
      overfit_counter = 0

  if overfit_counter > 3:
      print("Model is overfitting!")
  return overfit_counter


def stagnation(results, epoch:int,best_loss, epochs_no_imp):

  """
  A helper function that detects if the model's training has become
  stagnant

  Args:
    results: A dictionary contains keys of "train_loss", "train_acc",
    "val_loss" and "val_acc" each value is a list appended each epoch
    epoch: The amount of times the data is trained for
    best_loss: The lowest loss the model has detected
    epochs_no_imp: The amount of epochs that the model has no improved for

  Returns:
    best_loss: The new best loss
    epochs_no_imp: The new best epochs_no_imp
  """


  current_loss = results["val_loss"][epoch]
  min_delta = 1e-4
  if current_loss < best_loss - min_delta:
      best_loss = current_loss
      epochs_no_imp = 0
  else:
      epochs_no_imp += 1

  return best_loss, epochs_no_imp

def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "val_loss": [...],
             "val_acc": [...]}
    """
    loss = results["train_loss"]
    val_loss = results["val_loss"]

    accuracy = results["train_acc"]
    val_accuracy = results["val_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
