from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from torchvision.datasets.folder import ImageFolder
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import torch
class CustomDataset(Dataset):
    def __init__(self, folder_path1,folder_path2):
        self.folder_path1 = folder_path1
        self.folder_path2 = folder_path2
        #self.transform = self.get_simclr_pipeline_transform
        self.image_files1 = [f for f in os.listdir(folder_path1) if os.path.isfile(os.path.join(folder_path1, f))]
        self.image_files2 = [f for f in os.listdir(folder_path2) if os.path.isfile(os.path.join(folder_path2, f))]

        len1 = len(self.image_files1)
        len2 = len(self.image_files2)
        min_len = min(len1, len2)
        if len1 > len2:
            indices_to_keep = random.sample(range(len1), min_len)
            self.image_files1 = [self.image_files1[i] for i in indices_to_keep]

        # 如果list2更长，从list2中随机丢弃元素
        elif len2 > len1:
            indices_to_keep = random.sample(range(len2), min_len)
            self.image_files2 = [self.image_files2[i] for i in indices_to_keep]

        self.file_list = self.image_files1 + self.image_files2

        self.transform = transforms.Compose([transforms.RandomGrayscale(p=0.2),
                                                    GaussianBlur(kernel_size=int(16)),
                                                    transforms.ToTensor()])#transforms.ToTensor()会进行图片最终的resize

    def __len__(self):
        return len(self.image_files1)
    def __getitem__(self, idx):
        img_name1 = os.path.join(self.folder_path1, self.image_files1[idx])
        image1 = Image.open(img_name1)
        if self.transform:
            image1 = self.transform(image1)
        img_name2 = os.path.join(self.folder_path2, self.image_files2[idx])
        image2 = Image.open(img_name2)
        if self.transform:
            image2 = self.transform(image2)
        
        #print(torch.cat([image1,image2], dim=0).shape) 
        return image1,image2
    #torch.cat([image1.unsqueeze(0), image2.unsqueeze(0)], dim=0)
        #return image1,image2
# class ContrastiveLearningDataset:
#     def __init__(self, root_folder):
#         self.root_folder = root_folder

#     @staticmethod
#     def get_simclr_pipeline_transform(size, s=1):
#         """Return a set of data augmentation transformations as described in the SimCLR paper."""
#         color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
#         data_transforms = transforms.Compose([#transforms.RandomResizedCrop(size=size),
#                                               transforms.RandomHorizontalFlip(),
#                                               transforms.RandomApply([color_jitter], p=0.8),
#                                               transforms.RandomGrayscale(p=0.2),
#                                               GaussianBlur(kernel_size=int(0.1 * size)),
#                                               transforms.ToTensor()])
#         return data_transforms

#     def get_dataset(self, name, n_views):
#         valid_datasets = {  'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
#                                                               transform=ContrastiveLearningViewGenerator(
#                                                                   self.get_simclr_pipeline_transform(32),
#                                                                   n_views),
#                                                               download=True),

#                             'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
#                                                           transform=ContrastiveLearningViewGenerator(
#                                                               self.get_simclr_pipeline_transform(96),
#                                                               n_views),
#                                                           download=True),
#                             'custom_dataset': lambda: ImageFolder(root=self.root_folder,
#                                                                 transform=ContrastiveLearningViewGenerator(
#                                                                     self.get_simclr_pipeline_transform(32),
#                                                                     n_views))}

#         try:
#             dataset_fn = valid_datasets[name]
#         except KeyError:
#             raise InvalidDatasetSelection()
#         else:
#             return dataset_fn()
