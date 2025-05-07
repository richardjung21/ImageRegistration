import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import glob
import cv2
import torch
import xml.etree.ElementTree as ET
import numpy as np
import random

class OAI(Dataset):
    def __init__(self, root_dir: str, transform=None, split='train', regression=False):
        self.root_dir = root_dir
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
            ])
        else:
            self.transform = transform
        
        MONTHS = sorted(glob.glob(os.path.join(root_dir, "Month*")))
        self.images = []
        self.labels = []
        self.regression = regression

        for month in MONTHS:
            self.images.extend(sorted(glob.glob(os.path.join(month, "Images/*.jpg"))))
            self.labels.extend(sorted(glob.glob(os.path.join(month, "Labels/*.xml"))))

        self.all_images = self.images
        self.all_labels = {i: ET.parse(lbl).getroot() for i, lbl in enumerate(self.labels)}

        if split == 'train':
            self.images = self.images[:int(len(self.images) * 0.8)]
            self.labels = self.labels[:int(len(self.labels) * 0.8)]
        elif split == 'val':
            self.images = self.images[int(len(self.images) * 0.8):int(len(self.images) * 0.9)]
            self.labels = self.labels[int(len(self.labels) * 0.8):int(len(self.labels) * 0.9)]
        elif split == 'test':
            self.images = self.images[int(len(self.images) * 0.9):]
            self.labels = self.labels[int(len(self.labels) * 0.9):]
        
        # Preload labels into memory
        self.label_dict = {i: ET.parse(lbl).getroot() for i, lbl in enumerate(self.labels)}
    
    def __len__(self) -> int:
        return len(self.images)
    
    def get_random_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        idx = random.randrange(len(self.all_images))

        # Find matching pair idx of the first xray
        months = ["00", "12", "24", "36", "48"]
        first_month = self.all_images[idx].split('.')[-2][-2:]
        months.remove(first_month)
        second_month = months[random.randrange(len(months))]
        img2_path = self.all_images[idx].replace("Month_"+first_month, "Month_"+second_month).replace(first_month+".jpg", second_month+".jpg")
        # I have no freaking idea why but printing these suddenly makes the model stop working
        print(self.all_images[idx])
        print(img2_path)
        idx2 = self.all_images.index(img2_path)

        img1 = cv2.imread(self.all_images[idx])
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1 = torch.tensor(img1).permute(2, 0, 1) / 255

        img2 = cv2.imread(self.all_images[idx2])
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2 = torch.tensor(img2).permute(2, 0, 1) / 255

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        img1 = img1[None, :]
        img2 = img2[None, :]

        root1 = self.all_labels[idx]
        root2 = self.all_labels[idx2]
        lbl1 = int(root1.find('KL_Grade').text)
        lbl2 = int(root2.find('KL_Grade').text)

        return img1, lbl1, img2, lbl2
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image).permute(2, 0, 1) / 255
        
        if self.transform:
            image = self.transform(image)
        
        # Parse label from memory
        root = self.label_dict[idx]
        label_idx = int(root.find('KL_Grade').text)
        if self.regression:
            label = torch.tensor(label_idx).float()
        else:
            label = torch.zeros((5))
            label[label_idx] = 1
        
        return image, label