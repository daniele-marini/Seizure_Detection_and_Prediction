import torch
import numpy as np
import torch.nn as nn
from typing import Callable, Dict, Tuple
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from loss import focal_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopper:
  """
  Represent an object that is able to tell when the train need to stop.
  Needs to be put inside the train and keep updated, when early_stop returns
  true we need to stop the train
  """
  def __init__(self, patience : int = 1, min_delta : int = 0):
      self.patience = patience
      self.min_delta = min_delta
      self.counter = 0
      self.min_validation_loss = np.inf

  def early_stop(self, validation_loss : float) -> bool:
      if validation_loss < self.min_validation_loss:
          self.min_validation_loss = validation_loss
          self.counter = 0
      elif validation_loss > (self.min_validation_loss + self.min_delta):
          self.counter += 1
          if self.counter >= self.patience:
              return True
      return False

def get_correct_samples(scores: torch.Tensor, labels: torch.Tensor) -> int:

    """Get the number of correctly classified examples.

    Args:
        scores: the probability distribution.
        labels: the class labels.

    Returns: :return: the number of correct samples

    """
    classes_predicted = torch.argmax(scores, 1)
    return (classes_predicted == labels).sum().item()

def train(model: nn.Module,
          train_loader: DataLoader,
          device: torch.device,
          optimizer: torch.optim,
          criterion: Callable[[torch.Tensor, torch.Tensor], float],
          log_interval: int,
          epoch: int) -> Tuple[float, float]:

    """Train loop to train a neural network for one epoch.

    Args:
        model: the model to train.
        train_loader: the data loader containing the training data.
        device: the device to use to train the model.
        optimizer: the optimizer to use to train the model.
        criterion: the loss to optimize.
        log_interval: the log interval.
        epoch: the number of the current epoch

    Returns:
        the Cross Entropy Loss value on the training data,
        the accuracy on the training data.
    """
    correct = 0
    samples_train = 0
    loss_train = 0
    size_ds_train = len(train_loader.dataset)
    num_batches = len(train_loader)

    # start training
    model.train()
    for idx_batch, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        images = images.float()

        scores = model(images)

        loss = focal_loss(scores,labels)

        loss_train += loss.item() * len(images)
        samples_train += len(images)

        loss.backward()
        optimizer.step()
        correct += get_correct_samples(scores, labels)

        if log_interval > 0:
            if idx_batch % log_interval == 0:
                running_loss = loss_train / samples_train
                global_step = idx_batch + (epoch * num_batches)

    loss_train /= samples_train
    accuracy_training = 100. * correct / samples_train
    return loss_train, accuracy_training

def training_loop(num_epochs: int,
                  optimizer: torch.optim,
                  log_interval: int,
                  model: nn.Module,
                  loader_train: DataLoader,
                  early_stopping,
                  verbose: bool=True)->Dict:
    """Executes the training loop.

        Args:
            name_exp: the name for the experiment.
            num_epochs: the number of epochs.
            optimizer: the optimizer to use.
            log_interval: intervall to print on tensorboard.
            model: the mode to train.
            loader_train: the data loader containing the training data.
            loader_val: the data loader containing the validation data.
            verbose:

        Returns:
            A dictionary with the statistics computed during the train:
            the values for the train loss for each epoch
            the values for the train accuracy for each epoch
            the values for the validation accuracy for each epoch
            the time of execution in seconds for the entire loop
        """
    class_weights = torch.tensor([.3,10.,100.])
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight = class_weights)

    #class_weights = [1.0,10.0,100.0]
    #criterion = WeightedCrossEntropyLoss(weights=class_weights)
    loop_start = timer()

    losses_values = []

    for epoch in range(1, num_epochs + 1):
        time_start = timer()
        loss_train, accuracy_train = train(model, loader_train, device,
                                           optimizer, criterion, log_interval,
                                           epoch)

        time_end = timer()

        losses_values.append(loss_train)

        lr =  optimizer.param_groups[0]['lr']

        if verbose:
            print(f'Epoch: {epoch} '
                  f' Lr: {lr:.5f} '
                  f' Loss: Train = [{loss_train:.4f}]'
                  f' Accuracy: Train = [{accuracy_train:.2f}%]'
                  f' Time one epoch (s): {(time_end - time_start):.4f} ')


        if early_stopping is not None:
          if early_stopping.early_stop(loss_train):
            print(f'--- Early Stopping ---')
            break

    loop_end = timer()
    time_loop = loop_end - loop_start
    if verbose:
        print(f'Time for {num_epochs} epochs (s): {(time_loop):.3f}')

    return {'loss_values': losses_values,
            'time': time_loop}