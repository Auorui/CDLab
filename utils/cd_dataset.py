"""
2 num_classes
# CLCD dataset
    - train
        - label           label image in training set
        - image1               image at t1 in training set
        - image2               image at t2 in training set
    - val
    - test
"""
import cv2
import os
import torch
import numpy as np
from pyzjr.data import BaseDataset
from torch.utils.data import Dataset, DataLoader

clcp_map = {
            'NotChanged': np.array([0, 0, 0]),  # label 0   (rgb)
            'Changed': np.array([255, 255, 255]),  # label 1   (rgb)
        }

crop_scd_map = {
                0: np.array([255, 255, 255]),       # No Cropland Change
                1: np.array([0, 0, 255]),           # Water
                2: np.array([0, 100, 0]),           # Forest
                3: np.array([0, 128, 0]),           # Plantation
                4: np.array([0, 255, 0]),           # Grassland
                5: np.array([128, 0, 0]),           # Impervious surface
                6: np.array([0, 255, 255]),         # Road
                7: np.array([255, 0, 0]),           # Greenhouse
                8: np.array([255, 192, 0]),         # Bare soil
            }

class CLCDataset(BaseDataset):
    def __init__(
            self,
            root_dir,
            target_shape,
            num_classes=2,
            mode='train',
            dir_n1='image1',
            dir_n2='image2',
            color_map=clcp_map
    ):
        super(CLCDataset, self).__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.target_shape = self.to_2tuple(target_shape)
        data_dir = os.path.join(root_dir, self.mode)
        self.t1 = os.path.join(data_dir, dir_n1)
        self.t2 = os.path.join(data_dir, dir_n2)
        self.label = os.path.join(data_dir, 'label')
        self.image_name_list = self.SearchFileName(self.t1, ('.png', '.jpg', '.tif'))
        self.color_map = color_map
        self.disable_cv2_multithreading()

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, item):
        img_name = self.image_name_list[item]
        # normalize to [-1, 1]
        img1 = self.read_image(os.path.join(self.t1, img_name), to_rgb=True, normalize=True) * 2 - 1
        img2 = self.read_image(os.path.join(self.t2, img_name), to_rgb=True, normalize=True) * 2 - 1
        image_label = cv2.imread(os.path.join(self.label, img_name))
        rgb_image_label = cv2.cvtColor(image_label, cv2.COLOR_BGR2RGB)
        if self.mode == 'train':
            [img1, img2, imglabel] = self.augment([img1, img2, rgb_image_label], self.target_shape, prob=.5)
        else:
            [img1, img2, imglabel] = self.align([img1, img2, rgb_image_label], self.target_shape)
        img1 = torch.from_numpy(self.hwc2chw(img1)).float()
        img2 = torch.from_numpy(self.hwc2chw(img2)).float()
        imglabel = torch.from_numpy(self.process_label(imglabel)).long()
        return img1, img2, imglabel

    def process_label(self, rgb_image_label):
        label_seg = np.zeros(rgb_image_label.shape[:2], dtype=np.uint8)
        if self.color_map:
            label_seg[
                np.all(rgb_image_label == np.array(self.color_map['NotChanged']), axis=-1)
            ] = 0
            label_seg[
                np.all(rgb_image_label == np.array(self.color_map['Changed']), axis=-1)
            ] = 1
        else:
            label_seg = cv2.cvtColor(rgb_image_label, cv2.COLOR_RGB2GRAY)
        return label_seg


class CropSCDataset(BaseDataset):
    def __init__(
            self,
            root_dir,
            target_shape,
            num_classes=9,
            mode='train',
            dir_n1='im1',
            dir_n2='im2',
            color_map=None    # 标签已经是直接可用的
    ):
        super(CropSCDataset, self).__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.target_shape = self.to_2tuple(target_shape)
        data_dir = os.path.join(root_dir, self.mode)
        self.t1 = os.path.join(data_dir, dir_n1)
        self.t2 = os.path.join(data_dir, dir_n2)
        self.label = os.path.join(data_dir, 'label')
        self.image_name_list = self.SearchFileName(self.t1, ('.png', '.jpg', '.tif'))
        self.color_map = color_map
        self.disable_cv2_multithreading()

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, item):
        img_name = self.image_name_list[item]
        # normalize to [-1, 1]
        img1 = self.read_image(os.path.join(self.t1, img_name), to_rgb=True, normalize=True) * 2 - 1
        img2 = self.read_image(os.path.join(self.t2, img_name), to_rgb=True, normalize=True) * 2 - 1
        image_label = cv2.imread(os.path.join(self.label, img_name))
        rgb_image_label = cv2.cvtColor(image_label, cv2.COLOR_BGR2RGB)
        if self.mode == 'train':
            [img1, img2, imglabel] = self.augment([img1, img2, rgb_image_label], self.target_shape, prob=.5)
        else:
            [img1, img2, imglabel] = self.align([img1, img2, rgb_image_label], self.target_shape)
        img1 = torch.from_numpy(self.hwc2chw(img1)).float()
        img2 = torch.from_numpy(self.hwc2chw(img2)).float()
        imglabel = torch.from_numpy(self.process_label(imglabel)).long()
        return img1, img2, imglabel

    def process_label(self, rgb_image_label):
        if self.color_map:
            h, w, _ = rgb_image_label.shape
            label_seg = np.zeros((h, w), dtype=np.uint8)

            for idx, (_, color) in enumerate(self.color_map.items()):
                mask = np.all(rgb_image_label == color, axis=-1)
                label_seg[mask] = idx
        else:
            label_seg = cv2.cvtColor(rgb_image_label, cv2.COLOR_RGB2GRAY)
        return label_seg

def build_dataset(config, mode):
    conf = config.data
    if conf.num_classes == 2:
        dataset = CLCDataset(
            conf.dataset_path, conf.target_shape, 2,
            mode, conf.dir_n1, conf.dir_n2, conf.color_map)
    else:
        dataset = CropSCDataset(
            conf.dataset_path, conf.target_shape, conf.num_classes,
            mode, conf.dir_n1, conf.dir_n2, conf.color_map)
    return dataset


if __name__ == "__main__":
    # train_set = CLCDataset(r'E:\PythonProject\CDLab\data\CLCD_s', 512)
    # train_set = CropSCDataset(r'E:\PythonProject\CDLab\data\CropSCD_s', 512)
    train_set = build_dataset(r'E:\PythonProject\CDLab\config\dataset_config\CropSCD.yaml', 'train')
    train_loader = DataLoader(dataset=train_set, batch_size=4, num_workers=2, shuffle=True)

    for i, (image1, image2, label) in enumerate(train_loader):
        print(image1.shape, image2.shape, label.shape)
        # label_indices = torch.argmax(label, dim=1)
        print(f"Unique values in batch {i}: {torch.unique(label)}")
        # pyzjr.display(image1)
        # pyzjr.display(image2)