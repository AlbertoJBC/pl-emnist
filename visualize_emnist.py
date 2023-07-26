import os
import numpy as np
import matplotlib.pyplot as plt
from torch import optim, nn, utils, Tensor
from torchvision.datasets import EMNIST
from torchvision.transforms import ToTensor

# (Down)Load the dataset
dataset = EMNIST(os.getcwd(), split='balanced', download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset, shuffle=True)

# Save the classes
classes = dataset.classes
print('Total classes: ' + str(len(classes)))

# Plot example images
for image, label in train_loader:
    org_shape = image.shape
    plt.imshow(image.squeeze().T, cmap='gray') # For some reason, the original images are rotated
    plt.axis('off')
    plt.title('This is supposed to be a/an: ' + classes[label])
    plt.show(block=False)
    # Require user interaction
    try:
        next_img = input('Do you want to see another image? (Yes -> 1, No -> 0)\n')
        while not next_img in ['0','1']:
            print('Please, enter 0 or 1.\n')
            next_img = input('Do you want to see another image? (Yes -> 1, No -> 0)\n')
        if not int(next_img): break
    except KeyboardInterrupt:
        print('Ctrl+C -> Exiting the program...')
        break