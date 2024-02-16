import torch
import torch.nn.functional as F

def focal_loss(inputs, targets, gamma=5.0):

    label = targets.item()

    if label==0:
      alpha = 0.2
    elif label==1:
      alpha = 7
    elif label==2:
      alpha = 3

    # Compute the cross entropy loss
    CE_loss = F.cross_entropy(inputs, targets, reduction='none')
    # Calculate pt
    pt = torch.exp(-CE_loss)
    # Compute the focal loss
    focal_loss = alpha * (1 - pt) ** gamma * CE_loss

    return focal_loss