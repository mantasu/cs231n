from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import random
import torch

def compute_train_transform(seed=123456):
    """
    This function returns a composition of data augmentations to a single training image.
    Complete the following lines. Hint: look at available functions in torchvision.transforms
    """
    random.seed(seed)
    torch.random.manual_seed(seed)
    
    # Transformation that applies color jitter with brightness=0.4, contrast=0.4, saturation=0.4, and hue=0.1
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
    
    train_transform = transforms.Compose([
        ##############################################################################
        # TODO: Start of your code.                                                  #
        #                                                                            #
        # Hint: Check out transformation functions defined in torchvision.transforms #
        # The first operation is filled out for you as an example.
        ##############################################################################
        # Step 1: Randomly resize and crop to 32x32.
        transforms.RandomResizedCrop(32),
        # Step 2: Horizontally flip the image with probability 0.5
        transforms.RandomHorizontalFlip(p=0.5),
        # Step 3: With a probability of 0.8, apply color jitter (you can use "color_jitter" defined above.
        transforms.RandomApply([color_jitter], p=0.8),
        # Step 4: With a probability of 0.2, convert the image to grayscale
        transforms.RandomGrayscale(p=0.2),
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return train_transform
    
def compute_test_transform():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return test_transform


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        x_i = None
        x_j = None

        if self.transform is not None:
            ##############################################################################
            # TODO: Start of your code.                                                  #
            #                                                                            #
            # Apply self.transform to the image to produce x_i and x_j in the paper #
            ##############################################################################
            
            x_i = self.transform(img)
            x_j = self.transform(img)
            
            ##############################################################################
            #                               END OF YOUR CODE                             #
            ##############################################################################

        if self.target_transform is not None:
            target = self.target_transform(target)

        return x_i, x_j, target