import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
from dataset import CUFED
import torch.optim as optim
from utils import AP_partial
from torch.utils.data import DataLoader
from model import ModelGCNConcAfter as Model
from options.train_options import TrainOptions

args = TrainOptions().parse()


class EarlyStopper:
    def __init__(self, patience, min_delta, threshold):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_map = -float('inf')
        self.threshold = threshold

    def early_stop(self, validation_mAP):
        if validation_mAP >= self.threshold:
            return True, True
        if validation_mAP > self.max_validation_map:
            self.max_validation_map = validation_mAP
            self.counter = 0
            return False, True
        if validation_mAP < (self.max_validation_map - self.min_delta):
            self.counter += 1
            if self.counter > self.patience:
                return True, False
        return False, False


def train(model, loader, crit, opt, sched, device):
    model.train()
    epoch_loss = 0
    for batch in loader:
        feats_local, feat_global, label = batch
        feats_local = feats_local.to(device)
        feat_global = feat_global.to(device)
        label = label.to(device)
        opt.zero_grad()
        out_data = model(feats_local, feat_global, device)
        loss = crit(out_data, label)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    sched.step()
    return epoch_loss / len(loader)


def validate(model, dataset, loader, device):
    model.eval()
    gidx = 0
    scores = np.zeros((len(dataset), dataset.NUM_CLASS), dtype=np.float32)
    with torch.no_grad():
        for batch in loader:
            feats_local, feats_global, _, _ = batch
            feats_local = feats_local.to(device)
            feats_global = feats_global.to(device)
            out_data = model(feats_local, feats_global, device)
            shape = out_data.shape[0]
            scores[gidx:gidx+shape, :] = out_data.cpu()
            gidx += shape
    map_macro = AP_partial(dataset.labels, scores)[2]
    return map_macro


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if args.dataset == 'cufed':
        train_dataset = CUFED(root_dir=args.dataset_root, feats_dir=args.feats_dir, split_dir=args.split_dir)
        val_dataset = CUFED(args.dataset_root, feats_dir=args.feats_dir, split_dir=args.split_dir, is_train=False)
    else:
        sys.exit("Unknown dataset!")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    if args.verbose:
        print("running on {}".format(device))
        print("train_set = {}".format(len(train_dataset)))
        print("val_set = {}".format(len(val_dataset)))

    model = Model(args.gcn_layers, train_dataset.NUM_FEATS, train_dataset.NUM_CLASS).to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=args.milestones)
    early_stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta, threshold=args.threshold)

    start_epoch = 0
    if args.resume:
        data = torch.load(args.resume, map_location=device)
        start_epoch = data['epoch']
        model.load_state_dict(data['model_state_dict'], strict=True)
        opt.load_state_dict(data['opt_state_dict'])
        sched.load_state_dict(data['sched_state_dict'])
        if args.verbose:
            print("resuming from epoch {}".format(start_epoch))

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

        torch.save(model_config, os.path.join(args.save_dir, 'last_vigat_{}.pt'.format(args.dataset)))

        if is_save_ckpt:
            torch.save(model_config, os.path.join(args.save_dir, 'best_vigat_{}.pt'.format(args.dataset)))

        if is_early_stopping:
            print('Early stop at epoch {}'.format(epoch_cnt)) 
            break

        if args.verbose:
            print("[epoch {}] train_loss={} val_map={} dt_train={:.2f}sec dt_val={:.2f}sec dt={:.2f}sec".format(epoch_cnt, train_loss, val_map, t1 - t0, t3 - t2, t1 - t0 + t3 - t2))


if __name__ == '__main__':
    main()