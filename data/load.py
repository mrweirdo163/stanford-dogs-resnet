import os

from torchvision import transforms, datasets
from data.stanford_dogs_data import dogs
import configs

def load_datasets(set_name, input_size = 224):
    if set_name == 'stanford_dogs':
        input_transforms = transforms.Compose([
            transforms.RandomResizedCrop(input_size, ratio=(1, 1.3)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        train_dataset = dogs(root=configs.imagesets,
                                train=True,
                                cropped=False,
                                transform=input_transforms,
                                download=True)
        
        test_dataset = dogs(root=configs.imagesets,
                                train=False,
                                cropped=False,
                                transform=input_transforms,
                                download=True)
        
        classes = train_dataset.classes

        print("Training set stats:")
        train_dataset.stats()
        print("Testing set stats:")
        test_dataset.stats()

    return train_dataset, test_dataset, classes