 
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv
import re

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex

#This dataset is specifically for BCSS-WSSS dataset and is written to be used with taming-transformers
#It returns a dict with keys: image, mask, cls_label, file_path_
#image: preprocessed image in [-1,1] range, CHW format (follows ImagePaths in base.py)
# mask and labels are used for segmentation tasks or conditioning for Image Synthesis
class BCSSWSSS_TrainDataset(Dataset):
    #CLASSES = ["TUM", "STR", "LYM", "NEC", "BACK"]
    def __init__(self, img_root="data/sub_BCSS_WSSS/training", transform=None):
        super().__init__()
        self.img_root = img_root
        self.img_paths = sorted(glob.glob(os.path.join(img_root,"*.png")))
        self.get_images_and_labels(img_root)   
        self.transform = transform 
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        class_label = self.cls_labels[index] # one-hot vector of length 4 (TUM, STR, LYM, NEC), and save as an array


        assert os.path.exists(self.img_root), f"image path: {self.img_root} does not exist"


        # --- Load image & mask ---
        img = np.array(Image.open(img_path).convert("RGB"))   #  RGB
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        sample_dict = {
            "image": img,
            "cls_label": class_label,                       
            "img_path_": os.path.basename(img_path)
        }
        return sample_dict

    
    def get_images_and_labels(self, img_root=None):
        self.img_paths = []
        self.cls_labels = []

        self.img_paths = glob.glob(os.path.join(img_root, "*.png"))
        for img_path in self.img_paths:
            term_split = re.split("\[|\]", img_path)
            cls_label = np.array([int(x) for x in term_split[1]])
            self.cls_labels.append(cls_label)
        

dataset = BCSSWSSS_TrainDataset(img_root="data/sub_BCSS_WSSS/training")
print(f"Dataset size: {len(dataset)} images")

train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# # Lấy 1 batch ra in thử
# batch = next(iter(train_loader))
# print("Batch image shape:", batch["image"].shape)
# print("Batch cls_label shape:", batch["cls_label"])
# print("Batch file paths:", batch["img_path_"])

print(train_loader.dataset[0])

     