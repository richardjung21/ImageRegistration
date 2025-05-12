import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import glob
import cv2
import torch
import xml.etree.ElementTree as ET
import numpy as np
import random
import SimpleITK as sitk

class OAI(Dataset):
    def __init__(self, root_dir: str, transform=None, split='train', regression=False, image_registration=True, data_len=None):
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
        self.image_registration = image_registration

        # Redefine how images and labels are called if image_registration is true
        if image_registration:
            unique_subjects = []
            unique_labels = []
            for image in self.images:
                stripped = image.split('/')[-1]
                subject = stripped.split('L' if 'L' in stripped else 'R')[0] + ('L' if 'L' in stripped else 'R')
                
                temp_images = [s for s in self.images if (subject in s and subject not in unique_subjects)]
                temp_labels = [s for s in self.labels if (subject in s and subject not in unique_labels)]

                if len(temp_images) > 3:
                    unique_subjects.append(tuple(temp_images[:2]))
                    unique_subjects.append(tuple(temp_images[2:4]))
                    unique_labels.append(tuple(temp_labels[:2]))
                    unique_labels.append(tuple(temp_labels[2:4]))
                else:
                    unique_subjects.append(temp_images[:2])
                    unique_labels.append(temp_labels[:2])

            self.images = unique_subjects
            self.labels = unique_labels

        # Apply split
        total_len = len(self.images)
        if split == 'train':
            self.images = self.images[:int(total_len * 0.8)]
            self.labels = self.labels[:int(total_len * 0.8)]
        elif split == 'val':
            self.images = self.images[int(total_len * 0.8):int(total_len * 0.9)]
            self.labels = self.labels[int(total_len * 0.8):int(total_len * 0.9)]
        elif split == 'test':
            self.images = self.images[int(total_len * 0.9):]
            self.labels = self.labels[int(total_len * 0.9):]
        
        # Apply data_len if specified
        if data_len is not None:
            self.images = self.images[:data_len]
            self.labels = self.labels[:data_len]
        
        # Preload labels into memory
        if self.image_registration:
            self.label_dict = {i: [ET.parse(lbl).getroot() for lbl in lbls] for i, lbls in enumerate(self.labels)}
        else:
            self.label_dict = {i: ET.parse(lbl).getroot() for i, lbl in enumerate(self.labels)}
        
        self.data_len = len(self.images)
    
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
        if self.image_registration:
            image1 = cv2.imread(self.images[idx][0])
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image1 = torch.tensor(image1).permute(2, 0, 1) / 255

            image2 = cv2.imread(self.images[idx][1])
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            image2 = torch.tensor(image2).permute(2, 0, 1) / 255
            
            if self.transform:
                image1 = self.transform(image1)
                image2 = self.transform(image2)

            images = (image1, image2)
            
            # Parse label from memory
            root1 = self.label_dict[idx][0]
            root2 = self.label_dict[idx][1]
            label_idx1 = int(root1.find('KL_Grade').text)
            label_idx2 = int(root2.find('KL_Grade').text)
            if self.regression:
                label1 = torch.tensor(label_idx1).float()
                label2 = torch.tensor(label_idx2).float()

            else:
                label1 = torch.zeros((5))
                label1[label_idx1] = 1
                label2 = torch.zeros((5))
                label2[label_idx2] = 1
            labels = (label1, label2)
            
            return images, labels
        else:
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