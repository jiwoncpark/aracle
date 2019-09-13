import os, sys
import numpy as np
import torch
from PIL import Image

class HMIDataset(object):
    def __init__(self, root, transforms,
                 imgs_dirname='X_images_res256',
                 masks_dirname='Y_masks_res256',
                 img_list_fname='X_filenames.txt',
                 masks_list_fname='Y_filenames.txt'):
        
        self.root = root
        self.images_dir = imgs_dirname
        self.masks_dir = masks_dirname
        self.transforms = transforms
        # Paths of directories containing images and masks 
        self.imgs_dir = os.path.join(root, imgs_dirname)
        self.masks_dir = os.path.join(root, masks_dirname)
        # Load the filenames of all image/mask files, sorting them to
        # ensure that they are aligned
        self.imgs_list = list(sorted(os.listdir(self.imgs_dir)))
        self.masks_list = list(sorted(os.listdir(self.masks_dir)))
        
    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.imgs_dir, self.imgs_list[idx])
        mask_path = os.path.join(self.masks_dir, self.masks_list[idx])
        img = np.load(img_path)
        img = np.stack([img, img, img], axis=2) # for RGB
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = np.load(mask_path)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None] # shape [H, W, n_class - 1]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs_list)