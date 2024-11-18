import argparse
import albumentations as A
from torch.utils.data import Dataset
from model.EHANet import EHANet
from trainer import Trainer
from utils.dataset import DATASET
from utils.norm_utils import *
from utils.BCEloss import DiceBCELoss
import torch

parameters = dict()
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=42, type=int, help='manual seed')
parser.add_argument('--batch_size', default=16, type=int, help='batch size for train')
parser.add_argument('--data_path', required=True, type=str, help='train or test')
parser.add_argument('--num_epoch', default=50, type=int, help='train epochs')
parser.add_argument('--train_log_path', required=True, type=str, help='log path')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--num_workers', default=0, type=int, help='num_workers')
parser.add_argument('--checkpoint_path', default='', type=str, help='checkpoint_path')
parameters['size'] = (256,256)
args = parser.parse_args()


torch.manual_seed(args.seed)


transform = A.Compose([
    A.Rotate(limit=35, p=0.3),
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
])


(train_img, train_mask, train_edge), (valid_img, valid_mask, valid_edge) = load_dataset(args.data_path) 
path_list_img, path_list_mask, path_list_edge = shuffling(train_img, train_mask, train_edge)


train_dataset = DATASET(path_list_img, path_list_mask, path_list_edge, parameters['size'], transform=transform)
valid_dataset = DATASET(valid_img, valid_mask, valid_edge, parameters['size'], transform=None)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers,
                                           shuffle=True,
                                           pin_memory = True)

val_loader = torch.utils.data.DataLoader(valid_dataset,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         shuffle=False,
                                         pin_memory = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EHANet().to(device)

model.load_encoder_weight()
for param in model.encoder.parameters():
    param.requires_grad = False

params_to_optimize = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(params_to_optimize, lr=parameters['lr'])

loss = DiceBCELoss()

training = Trainer(
        model=model,
        optimizer=optimizer,
        batch_size= args.batch_size,
        train_loader=train_loader,
        val_loader=val_loader,
        train_log_path= args.train_log_path,
        size = parameters["size"],
        device = device,
        checkpoint_path = args.checkpoint_path,
        num_epoch= args.num_epoch,
        lr = args.lr,
        loss_region=loss
    )

if __name__ == '__main__':
    training.train()