import os, sys
import numpy as np
import torch
from PIL import Image

class FasterRCNNDataset(object):
    def __init__(self, transforms,
                 imgs_dir='X_images_uncropped_circle_res256',
                 masks_dir='Y_masks_uncropped_circle_res256',):
        """

        Parameters
        ----------
        transforms : 
        imgs_dir : str or os.path object
            path to the directory containing the X images
        masks_dir : str or os.path. object
            path to the directory containing the Y masks

        """
        
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transforms = transforms
        # Load the filenames of all image/mask files, sorting them to
        # ensure that they are aligned
        self.imgs_list = list(sorted(os.listdir(self.imgs_dir)))
        self.masks_list = list(sorted(os.listdir(self.masks_dir)))
        assert len(self.imgs_list) == len(self.masks_list)

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
        masks = mask == obj_ids[:, None, None] # shape [N, H, W]
        masks = masks.astype(int)

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids) # this didn't work
        #num_objs = len(masks)

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

if __name__ == '__main__':
    import aracle.transforms_utils as transforms_utils
    import aracle.data

    transforms = transforms_utils.Compose([transforms_utils.ToTensor()])
    #data_dir = os.path.dirname(aracle.data.__file__)
    data_dir = '/nobackup/jpark45'
    imgs_dir = os.path.join(data_dir, 'minidata', 'X_images_uncropped_circle_res256')
    masks_dir = os.path.join(data_dir, 'minidata', 'Y_masks_uncropped_circle_res256')

    hmi_dataset = HMIDataset(transforms,
                             imgs_dir=imgs_dir,
                             masks_dir=masks_dir)
    n_data = len(hmi_dataset)
    print(n_data)
    print(hmi_dataset[0][0].shape)
    sample_dict = hmi_dataset[0][1]
    print(sample_dict['masks'].shape)
    print(sample_dict['boxes'].shape)
