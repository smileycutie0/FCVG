import os, io, csv, math, random
import numpy as np
from einops import rearrange

import torch
from decord import VideoReader
import json

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from copy import deepcopy
from PIL import Image


def pil_image_to_numpy(image):
    """Convert a PIL image to a NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)


def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images.float() / 255.


class MatchPoseDataset(Dataset):
    def __init__(
            self,
            dataset_path, 
            width=768,
            height=512,
            sample_n_frames=25,
            interval_frame=1,
            shuffle=True
        ):
        self.data_path_list = [os.path.join(dataset_path, v) for v in sorted(os.listdir(dataset_path))]
        if shuffle:
            random.shuffle(self.data_path_list)    
        self.length           = len(self.data_path_list)
        self.sample_n_frames  = sample_n_frames
        self.width            = width
        self.height           = height
        self.interval_frame   = interval_frame
        sample_size           = (height, width)

        self.pixel_transforms = transforms.Compose([
            transforms.Resize(sample_size, antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])


        def random_transpose(tensor: torch.tensor):
            if random.random() > 0.5:
                if len(tensor.shape) == 3:
                    if random.random() > 0.5:
                        return tensor
                    else:
                        tensor[[0,-1],...] = tensor[[-1,0],...]
                        return tensor
                elif len(tensor.shape) == 4:
                    if random.random() > 0.5:
                        return tensor
                    else:
                        tensor[:, [0,-1],...] = tensor[:, [-1,0],...]
                        return tensor
                else:
                    exit()
            else:
                return tensor
        self.pose_transforms = transforms.Compose([
            transforms.Lambda(random_transpose), 
            transforms.Resize(sample_size, antialias=True),
        ])

    
    def get_batch(self, idx):
        video_folder = os.path.join(self.data_path_list[idx], 'images')
        guides_folder = os.path.join(self.data_path_list[idx], 'fusion')

        videos_path = [os.path.join(video_folder, img) for img in sorted(os.listdir(video_folder))]
        guides_path = [os.path.join(guides_folder, img) for img in sorted(os.listdir(guides_folder))]
        
        images = np.array([pil_image_to_numpy(Image.open(video_path)) for video_path in videos_path])[:self.sample_n_frames]
        poses = np.array([pil_image_to_numpy(Image.open(guide_path)) for guide_path in guides_path])[:self.sample_n_frames]

        reference_image = np.array([images[0], images[-1]])
        pixel_values = numpy_to_pt(images)
        guide_values = numpy_to_pt(poses)
        reference_image = numpy_to_pt(reference_image)

        return pixel_values, guide_values, reference_image



    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pixel_values, guide_values, reference_image = self.get_batch(idx)

        pixel_values = self.pixel_transforms(pixel_values)

        reference_image = self.pixel_transforms(reference_image)[0]
        guide_values = self.pose_transforms(guide_values)

        sample = dict(
            pixel_values = pixel_values, 
            guide_values = guide_values,
            reference_image = reference_image, 
           )
        return sample

import cv2
def recover_batch(batch, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(os.path.join(folder_path, "pose")):
            os.makedirs(os.path.join(folder_path, "pose"))
        if not os.path.exists(os.path.join(folder_path, "rgb")):
            os.makedirs(os.path.join(folder_path, "rgb"))
        pixel_values = batch["pixel_values"]
        guide_values = batch["guide_values"]
        ref_values = batch["reference_image"]
        pixel_values = (((pixel_values + 1) / 2).numpy() * 255).astype(np.uint8).clip(min=0, max=255).transpose(0, 2, 3, 1)
        guide_values = (guide_values.numpy() * 255).astype(np.uint8).clip(min=0, max=255).transpose(0, 2, 3, 1)

        for idx in range(len(pixel_values)):
            frame = pixel_values[idx]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(folder_path, "rgb", "{}.png".format(idx)), frame)
        for idx in range(len(guide_values)):
            frame = guide_values[idx]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(folder_path, "pose", "{}.png".format(idx)), frame)

if __name__ == "__main__":

    dataset = MatchPoseDataset(
        dataset_path="/hdd/zty/code/2024work/svd_vfi_contronext/datasets/videos_frames25",
        interval_frame=1,
        sample_n_frames=25,
    )

    recover_batch(dataset[random.randint(0, 100)], "./saved_images/temp")
