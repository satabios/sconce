from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
import math
import torch
import os
def Dataset(which):
    if 'cifar10' in which:

        image_size = 512
        transforms = {
            # VGG
        
            "train": Compose([
                RandomCrop(image_size, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
            ]),
            "test": ToTensor(),} if('vgg' in which) else \
            {
                'train': Compose([
                            ToTensor(),
                            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize between -1 and 1
                        ]),
                'test': Compose([
                            ToTensor(),
                            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize between -1 and 1
                        ])
            }
        dataset = {}
        for split in ["train", "test"]:
            dataset[split] = CIFAR10(
                root="../../data/dataset/cifar10",
                train=(split == "train"),
                download=True,
                transform=transforms[split],
            )
        dataloader = {}
        for split in ['train', 'test']:
            dataloader[split] = DataLoader(
                dataset[split],
                batch_size=512,
                shuffle=(split == 'train'),
                num_workers=0,
                pin_memory=True,
                drop_last = True
            )
        return dataloader['test']
    elif which == 'imagenet':
        imagenet_data_path = '/home/sathya/Downloads/Imagenet/imagenet-val'
        batch_size = 128
        from torchvision import transforms
        def build_val_transform(size):
            return transforms.Compose([
                transforms.Resize(int(math.ceil(size / 0.875))),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

        val_data = ImageFolder(
            root=os.path.join(imagenet_data_path),
            transform=build_val_transform(224)
        )

        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        return val_loader
