import os
import torch
import torch.nn.functional as F
import numpy as np
import utils.visualization as visual
from utils import dataloader
from tqdm import tqdm
import random
from utils.Metrics import Evaluator
from utils.loss import FocalDiceLoss
from models.FullNet import FullNet
import time
import argparse
import torchvision.transforms as transforms
from PIL import Image

start_time = time.time()


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_pretrained_weights(model, pretrained_path, device):
    try:
        checkpoint = torch.load(pretrained_path, map_location=device)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        state_dict = {k.replace('layers.', 'encoder_stages.'): v for k, v in state_dict.items()}
        model_dict = model.state_dict()

        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    model_dict[k] = v

        model.load_state_dict(model_dict)
        return True
    except Exception as e:
        print(f"失败: {e}")
        return False


def train(train_loader,
          val_loader,
          Eva_train,
          Eva_val,
          vis,
          save_path,
          net,
          criterion,
          optimizer,
          num_epoches,
          epoch,
          lr_scheduler,
          trainsize,
          device):
    global best_iou

    net.train(True)
    epoch_loss = 0
    num_batches_train = len(train_loader)
    pbar_train = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epoches} [Training]")
    for i, (A, B, mask) in enumerate(pbar_train):
        A, B, Y = A.to(device), B.to(device), mask.to(device)
        optimizer.zero_grad()
        outputs = net(A, B)
        loss_main = criterion(outputs[0], Y)
        loss_aux1 = criterion(outputs[1], Y)
        loss_aux2 = criterion(outputs[2], Y)
        loss_aux3 = criterion(outputs[3], Y)
        loss = loss_main + 0.4 * loss_aux1 + 0.2 * loss_aux2 + 0.1 * loss_aux3
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        with torch.no_grad():
            output = F.sigmoid(outputs[0])
            pred_gpu = (output >= 0.5).int()
            Eva_train.add_batch(Y, pred_gpu)

    train_iou = Eva_train.Intersection_over_Union()[1]
    train_f1 = Eva_train.F1()[1]
    train_loss = epoch_loss / num_batches_train
    print(f"\n[Epoch {epoch} Training] Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, F1: {train_f1:.4f}")

    net.eval()
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epoches} [Validation]")
    for i, (A, B, mask, base_name, info) in enumerate(val_pbar):
        with torch.no_grad():
            A, B, Y = A.to(device), B.to(device), mask.to(device)
            output_tile = F.sigmoid(net(A, B)[0])
            pred_to_eval_gpu = (output_tile >= 0.5).int()
            Eva_val.add_batch(Y, pred_to_eval_gpu)

    val_iou = Eva_val.Intersection_over_Union()[1]
    val_pre = Eva_val.Precision()[1]
    val_recall = Eva_val.Recall()[1]
    val_f1 = Eva_val.F1()[1]
    vis.add_scalar(epoch, val_iou, 'Validation/mIoU')
    vis.add_scalar(epoch, val_f1, 'Validation/F1')
    print(
        f"[Epoch {epoch} Validation] mIoU: {val_iou:.4f}, Precision: {val_pre:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

    if val_iou >= best_iou:
        best_iou = val_iou
        print(f"*** New Best mIoU: {best_iou:.4f}! Saving best model for epoch {epoch}... ***")
        torch.save(
            {'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
             'best_iou': best_iou, 'lr_scheduler_state_dict': lr_scheduler.state_dict()},
            os.path.join(save_path, '_best_iou.pth'))

    latest_checkpoint_path = os.path.join(save_path, 'checkpoint_latest.pth')
    torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou, 'lr_scheduler_state_dict': lr_scheduler.state_dict()}, latest_checkpoint_path)


if __name__ == '__main__':
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--trainsize', type=int, default=256)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--data_name', type=str, default='LEVIR')
    parser.add_argument('--model_name', type=str, default='FullNet')
    parser.add_argument('--base_save_path', type=str, default='/root/autodl-tmp/data/LEVIR/')
    parser.add_argument('--resume', type=str, default=None)
    opt = parser.parse_args()

    device = torch.device(f'cuda:{opt.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    opt.save_path = os.path.join(opt.base_save_path, opt.data_name, opt.model_name)

    if opt.data_name == 'LEVIR':
        opt.train_root = '/root/autodl-tmp/data/LEVIR/train/'
        opt.val_root = '/root/autodl-tmp/data/LEVIR/test/'
    elif opt.data_name == 'SYSU':
        opt.train_root = "/root/autodl-tmp/data/SYSU/train/"
        opt.val_root = "/root/autodl-tmp/data/SYSU/test/"
    elif opt.data_name == 'WHU':
        opt.train_root = '/root/autodl-tmp/data/WHU/train512/'
        opt.val_root = '/root/autodl-tmp/data/WHU/test512/'
    elif opt.data_name == 'CDD':
        opt.train_root = '/root/autodl-tmp/data/CDD/Real/train'
        opt.val_root = '/root/autodl-tmp/data/CDD/Real/test'
    elif opt.data_name == 'CLCD':
        opt.train_root = '/root/autodl-tmp/data/CLCD/train'
        opt.val_root = '/root/autodl-tmp/data/CLCD/test'

    train_loader = dataloader.get_loader(opt.train_root, opt.batchsize, opt.trainsize, num_workers=12, shuffle=True,
                                         pin_memory=True)
    val_loader = dataloader.get_test_loader(opt.val_root, opt.batchsize, opt.trainsize, num_workers=12, shuffle=False,
                                            pin_memory=True)

    Eva_train = Evaluator(num_class=2, device=device)
    Eva_val = Evaluator(num_class=2, device=device)

    model = FullNet(img_size=[256, 256], in_channels=3, num_classes=1, dims=[96, 192, 384, 768], depths=[2, 2, 8, 2],
                    deep_supervision=True, s2fm_r1=8, s2fm_r2=8, alpha_inits=[0, 0, 0, 0]).to(device)

    criterion = FocalDiceLoss(focal_weight=0.8, dice_weight=0.2).to(device)
    alpha_param_ids = list(map(id, model.alphas))
    base_params = filter(lambda p: id(p) not in alpha_param_ids, model.parameters())

    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': opt.lr, 'weight_decay': 0.0025},
        {'params': model.alphas, 'lr': 1e-2, 'weight_decay': 0.0}
    ])

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    start_epoch = 1
    best_iou = 0.0
    latest_resume_path = os.path.join(opt.save_path, 'checkpoint_latest.pth')
    best_iou_resume_path = os.path.join(opt.save_path, '_best_iou.pth')
    if opt.resume is None:
        if os.path.isfile(latest_resume_path):
            opt.resume = latest_resume_path
        elif os.path.isfile(best_iou_resume_path):
            opt.resume = best_iou_resume_path

    if opt.resume and os.path.isfile(opt.resume):
        checkpoint = torch.load(opt.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint.get('best_iou', 0.0)
        print(f"=> Checkpoint loaded! Resuming from Epoch {start_epoch}.")

    vis = visual.Visualization()
    vis.create_summary(opt.save_path)

    for epoch in range(start_epoch, opt.epoch + 1):
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        Eva_train.reset()
        Eva_val.reset()
        train(train_loader, val_loader, Eva_train, Eva_val, vis, opt.save_path, model, criterion, optimizer, opt.epoch,
              epoch, lr_scheduler, opt.trainsize, device)
        lr_scheduler.step()

    vis.close_summary()
    end_time = time.time()
    print(f'Total training time: {(end_time - start_time) / 3600:.2f} hours')