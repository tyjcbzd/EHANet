import datetime
import time
from torch.utils.tensorboard import SummaryWriter
from utils.Softmax_loss import lovasz_softmax
from utils.norm_utils import *
from utils.metric_utils import *

class Trainer(object):
    def __init__(self, model, optimizer, size, batch_size, train_log_path, train_loader, val_loader, device, checkpoint_path,
                 num_epoch, lr, loss_region):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epoch = num_epoch
        self.iteration = 0
        self.train_log_path = train_log_path
        self.size = size
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.model = model
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.lr = lr
        self.loss_region = loss_region
    def train(self):

        create_dir("files")

        if os.path.exists(self.train_log_path):
            print("Log file exists")
        else:
            train_log = open(self.train_log_path, "w")  
            train_log.write("\n")
            train_log.close()

        datetime_object = str(datetime.datetime.now())
        print_and_save(self.train_log_path, datetime_object)


        results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        writer = SummaryWriter('logs/babysitting')
        loss_name = "Lovas softmax"

        data_str = f"Hyperparameters:\nImage Size: {self.size}\nBatch Size: {self.batch_size}\nLR: {self.lr}\nEpochs: {self.num_epoch}\n"
        data_str += f"Optimizer: Adam\nLoss: {loss_name}\n"
        print_and_save(self.train_log_path, data_str)

        best_valid_loss = float('inf')
        start_time = time.time()
        for epoch in range(self.num_epoch):
            train_loss, train_edge_loss, train_out_loss = self.train_epoch()
            valid_loss, valid_edge_loss, valid_out_loss, dice_y, jac_y = self.evaluate()

            writer.add_scalar('Loss/train', train_loss, global_step=(epoch + 1))
            writer.add_scalar('Loss/valid', valid_loss, global_step=(epoch + 1))
            writer.add_scalar('Loss/train_edge_loss', train_edge_loss, global_step=(epoch + 1))
            writer.add_scalar('Loss/valid_edge_loss', valid_edge_loss, global_step=(epoch + 1))
            writer.add_scalar('Loss/train_out_loss', train_out_loss, global_step=(epoch + 1))
            writer.add_scalar('Loss/valid_out_loss', valid_out_loss, global_step=(epoch + 1))
            writer.add_scalar('out_mIOU/train', jac_y, global_step=(epoch + 1))
            writer.add_scalar('out_Dice/train', dice_y, global_step=(epoch + 1))

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                data_str = f"Saving checkpoint: {self.checkpoint_path}"
                print_and_save(self.train_log_path, data_str)
                torch.save(self.model.state_dict(), self.checkpoint_path)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            data_str = f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
            data_str += f'\t Train Loss: {train_loss:.3f}\n'
            data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
            data_str += f'\t train_edge_loss: {train_edge_loss:.3f}\n'
            data_str += f'\t valid_edge_loss: {valid_edge_loss:.3f}\n'

            data_str += f'\t train_out_loss: {train_out_loss:.3f}\n'
            data_str += f'\t valid_out_loss: {valid_out_loss:.3f}\n'

            data_str += f'\t y_Dice: {dice_y:.3f}\n'
            data_str += f'\t y_mIOU: {jac_y:.3f}\n'

            print_and_save(self.train_log_path, data_str)


    def train_epoch(self):
        epoch_loss = 0
        self.model.train()

        for i, (image, target, edge) in enumerate(self.train_loader):
            image, target, edge = image.to(self.device), target.to(self.device), edge.to(self.device)
            self.optimizer.zero_grad()
            edge_pred, s1, s2, s3, s4, s_g = self.model(image)

            edge_loss = lovasz_softmax(edge_pred, edge)
            out_loss = (self.loss_region(s1,target)+
                        self.loss_region(s2,target)+
                        self.loss_region(s3,target)+
                        self.loss_region(s4,target)+
                        self.loss_region(s_g,target))
            two_loss = edge_loss + out_loss
            two_loss.backward()
            self.optimizer.step()

            epoch_loss += two_loss.item()

        epoch_loss = epoch_loss / len(self.train_loader)
        return epoch_loss, edge_loss, out_loss


    def evaluate(self):
        epoch_loss = 0
        self.model.eval()
        with torch.no_grad():
            all_epoch_loss, all_edge_loss, all_out_loss, all_dice_y, all_jac_y =  [],[],[],[],[]
            for i, (image, target, edge) in enumerate(self.val_loader):
                image, target, edge= image.to(self.device), target.to(self.device), edge.to(self.device)

                edge_pred, s1, s2, s3, s4, s_g = self.model(image)
                y_pred = s_g
                edge_loss = lovasz_softmax(edge_pred, edge)
                out_loss = (self.loss_region(s1,target)+
                            self.loss_region(s2,target)+
                            self.loss_region(s3,target)+
                            self.loss_region(s4,target)+
                            self.loss_region(s_g,target))

                two_loss = edge_loss + out_loss
                epoch_loss += two_loss.item()

                y_true = target.cpu().numpy()
                y_pred = y_pred.cpu().numpy()

                y_pred = y_pred > 0.5
                y_pred = y_pred.reshape(-1)
                y_pred = y_pred.astype(np.uint8)

                y_true = y_true > 0.5
                y_true = y_true.reshape(-1)
                y_true = y_true.astype(np.uint8)

                dice_y = dice_score(y_true, y_pred)
                jac_y = miou_score(y_true, y_pred)
                
                all_edge_loss.append(edge_loss)
                all_out_loss.append(out_loss)
                all_epoch_loss.append(out_loss+edge_loss)
                all_dice_y.append(dice_y)
                all_jac_y.append(jac_y)
        epoch_loss = torch.sum(all_epoch_loss) / len(self.val_loader)
        edge_loss = torch.sum(all_edge_loss) / len(self.val_loader)
        out_loss = torch.sum(all_out_loss) / len(self.val_loader)
        dice_y = torch.sum(all_dice_y) / len(self.val_loader)
        jac_y = torch.sum(all_jac_y) / len(self.val_loader)
        return epoch_loss, edge_loss, out_loss, dice_y, jac_y


    def test(self):

        return None