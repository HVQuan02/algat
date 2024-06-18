import argparse
import time
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import CUFED
from utils import AP_partial
from model import ModelGCNConcAfter as Model

parser = argparse.ArgumentParser(description='GCN Album Classification')
parser.add_argument('--seed', type=int, default=2024, help='seed for randomness')
parser.add_argument('--gcn_layers', type=int, default=2, help='number of gcn layers')
parser.add_argument('--dataset', default='cufed', choices=['pec', 'cufed'])
parser.add_argument('--dataset_root', default='/kaggle/input/thesis-cufed/CUFED', help='dataset root directory')
parser.add_argument('--feats_dir', default='/kaggle/input/cufed-feats', help='global and local features directory')
parser.add_argument('--split_dir', default='/kaggle/input/cufed-full-split', help='train split and val split')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--milestones', nargs="+", type=int, default=[110, 160], help='milestones of learning decay')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
parser.add_argument('--resume', default=None, help='checkpoint to resume training')
parser.add_argument('--save_folder', default='weights', help='directory to save checkpoints')
parser.add_argument('--patience', type=int, default=30, help='patience of early stopping')
parser.add_argument('--min_delta', type=float, default=0.5, help='min delta of early stopping')
parser.add_argument('--threshold', type=float, default=95, help='val mAP threshold of early stopping')
parser.add_argument('-v', '--verbose', action='store_true', help='show details')
args = parser.parse_args()

class EarlyStopper:
    def __init__(self, patience, min_delta, threshold):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_mAP = -float('inf')
        self.threshold = threshold

    def early_stop(self, validation_mAP):
        if validation_mAP >= self.threshold:
            return True, True
        if validation_mAP > self.max_validation_mAP:
            self.max_validation_mAP = validation_mAP
            self.counter = 0
            return False, True
        if validation_mAP < (self.max_validation_mAP - self.min_delta):
            self.counter += 1
            if self.counter > self.patience:
                return True, False
        return False, False

def train(model, loader, crit, opt, sched, device):
    epoch_loss = 0
    for batch in loader:
        feats, feat_global, label = batch

        feats = feats.to(device)
        feat_global = feat_global.to(device)
        label = label.to(device)

        opt.zero_grad()
        out_data = model(feats, feat_global, device)
        loss = crit(out_data, label)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()

    sched.step()
    return epoch_loss / len(loader)

def validate(model, dataset, loader, device):
    scores = np.zeros((len(dataset), dataset.NUM_CLASS), dtype=np.float32)
    gidx = 0
    model.eval()
    with torch.no_grad():
        for feats, feat_global, _, _ in loader:
            feats = feats.to(device)
            feat_global = feat_global.to(device)
            out_data = model(feats, feat_global, device)
            shape = out_data.shape[0]
            scores[gidx:gidx+shape, :] = out_data.cpu()
            gidx += shape
    map_macro = AP_partial(dataset.labels, scores)[2]
    return map_macro

def main():
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    if args.dataset == 'cufed':
        train_dataset = CUFED(root_dir=args.dataset_root, feats_dir=args.feats_dir, split_dir=args.split_dir)
        val_dataset = CUFED(args.dataset_root, feats_dir=args.feats_dir, split_dir=args.split_dir, is_train=False)
    else:
        sys.exit("Unknown dataset!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    if args.verbose:
        print("running on {}".format(device))
        print("num of train set = {}".format(len(train_dataset)))
        print("num of val set = {}".format(len(val_dataset)))

    start_epoch = 0
    model = Model(args.gcn_layers, train_dataset.NUM_FEATS, train_dataset.NUM_CLASS).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.BCEWithLogitsLoss()
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=args.milestones)

    if args.resume:
        data = torch.load(args.resume)
        start_epoch = data['epoch']
        model.load_state_dict(data['model_state_dict'])
        opt.load_state_dict(data['opt_state_dict'])
        sched.load_state_dict(data['sched_state_dict'])
        if args.verbose:
            print("resuming from epoch {}".format(start_epoch))

    early_stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta, threshold=args.threshold)

    model.train()
    for epoch in range(start_epoch, args.num_epochs):
        epoch_cnt = epoch + 1
        
        t0 = time.perf_counter()
        train_loss = train(model, train_loader, crit, opt, sched, device)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        val_map = validate(model, val_dataset, val_loader, device)
        t3 = time.perf_counter()

        is_early_stopping, is_save_ckpt = early_stopper.early_stop(val_map)

        model_config = {
            'epoch': epoch_cnt,
            'loss': train_loss,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict(),
            'sched_state_dict': sched.state_dict()
        }

        torch.save(model_config, os.path.join(args.save_folder, 'last-vigat-{}.pt'.format(args.dataset)))

        if is_save_ckpt:
            torch.save(model_config, os.path.join(args.save_folder, 'best-vigat-{}.pt'.format(args.dataset)))

        if is_early_stopping:
            print('Stop at epoch {}'.format(epoch_cnt)) 
            break

        if args.verbose:
            print("[epoch {}] train_loss={} val_map={} dt_train={:.2f}sec dt_val={:.2f}sec dt={:.2f}sec".format(epoch_cnt, train_loss, val_map, t1 - t0, t3 - t2, t1 - t0 + t3 - t2))


if __name__ == '__main__':
    main()