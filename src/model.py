import torch
import torch.nn as nn
from torchvision.models import resnet50

def modified_resnet50(num_channels=5):

    model = resnet50(pretrained=True)

    original_first_layer = model.conv1
    new_first_layer = nn.Conv2d(in_channels=num_channels,
                                out_channels=original_first_layer.out_channels,
                                kernel_size=original_first_layer.kernel_size,
                                stride=original_first_layer.stride,
                                padding=original_first_layer.padding,
                                bias=original_first_layer.bias)

    with torch.no_grad():
        new_first_layer.weight[:, :3, :, :] = original_first_layer.weight
        for i in range(3, num_channels):
            new_first_layer.weight[:, i, :, :] = original_first_layer.weight[:, i % 3, :, :]

    # Replace the first convolutional layer in the model
    model.conv1 = new_first_layer
    # Replace the final classifier
    model.fc = nn.Linear(model.fc.in_features, 3)
    return model