from torch.utils.data import Dataset
import cv2
import numpy as np

class DATASET(Dataset):
    def __init__(self, images_path, masks_path, edges_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.edges_path = edges_path
        self.transform = transform
        self.n_samples = len(images_path)
        self.size = size

    def __getitem__(self, index):
        """ Image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        edge = cv2.imread(self.edges_path[index], cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask, edge=edge)
            image = augmentations["image"]
            mask = augmentations["mask"]
            edge = augmentations["edge"]

        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0
        image = image.astype(np.float32)

        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0
        mask = mask.astype(np.float32)

        edge = cv2.resize(edge, self.size)
        edge = np.expand_dims(edge, axis=0)
        edge = edge / 255.0
        edge = edge.astype(np.float32)

        return image, mask, edge

    def __len__(self):
        return self.n_samples


class PredGT(Dataset):
    def __init__(self, preds_path, gts_path):
        super().__init__()

        self.preds_path = preds_path
        self.gts_path = gts_path
        self.n_samples = len(preds_path)

    def __getitem__(self, index):
        """ Image """
        pred = cv2.imread(self.preds_path[index], cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(self.gts_path[index], cv2.IMREAD_GRAYSCALE)

        pred = pred/255.0
        pred = pred.astype(np.float32)

        gt = gt / 255.0
        gt = gt.astype(np.float32)

        return pred, gt

    def __len__(self):
        return self.n_samples
