import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import sys
from torch.utils.data import DataLoader
from datasets import CUFED
from utils import AP_partial, spearman_correlation, showCM
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, classification_report
from model import ModelGCNConcAfter as Model

parser = argparse.ArgumentParser(description='GCN Album Classification')
parser.add_argument('model', nargs=1, help='trained model')
parser.add_argument('--gcn_layers', type=int, default=2, help='number of gcn layers')
parser.add_argument('--dataset', default='cufed', choices=['pec', 'cufed'])
parser.add_argument('--dataset_root', default='/kaggle/input/thesis-cufed/CUFED', help='dataset root directory')
parser.add_argument('--feats_dir', default='/kaggle/input/cufed-feats', help='global and local features directory')
parser.add_argument('--split_dir', default='/kaggle/input/cufed-full-split', help='train split and val split')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
parser.add_argument('--save_scores', action='store_true', help='save the output scores')
parser.add_argument('--save_path', default='scores.txt', help='output path')
parser.add_argument('--threshold', type=float, default=0.75, help='threshold for logits to labels')
parser.add_argument('-v', '--verbose', action='store_true', help='show details')
args = parser.parse_args()

def evaluate(model, dataset, loader, out_file, device):
    scores = torch.zeros((len(dataset), dataset.NUM_CLASS), dtype=torch.float32)
    gidx = 0
    model.eval()
    importance_list = []
    frame_wid_list = []
    obj_wid_list = []

    with torch.no_grad():
        for batch in loader:
            feats, feat_global, _, importances = batch

            # Run model with all frames
            feats = feats.to(device)
            feat_global = feat_global.to(device)
            out_data, wids_objects, wids_frame_local, wids_frame_global = model(feats, feat_global, device, get_adj=True)
            shape = out_data.shape[0]
            
            if out_file:
                for j in range(shape):
                    video_name = dataset.videos[gidx + j]
                    out_file.write("{} ".format(video_name))
                    out_file.write(' '.join([str(x.item()) for x in out_data[j, :]]))
                    out_file.write('\n')

            scores[gidx:gidx+shape, :] = out_data.cpu()
            gidx += shape
            importance_list.append(importances)
            avg_frame_wid = (wids_frame_local + wids_frame_global) / 2
            frame_wid_list.append(torch.from_numpy(avg_frame_wid))
            obj_wid_list.append(torch.from_numpy(np.reshape(wids_objects.mean(axis=1), (shape, -1))))
    
    m = nn.Sigmoid()
    preds = m(scores)
    preds[preds >= args.threshold] = 1
    preds[preds < args.threshold] = 0
    scores, preds = scores.numpy(), preds.numpy()

    map_micro, map_macro = AP_partial(dataset.labels, scores)[1:3]

    acc = accuracy_score(dataset.labels, preds)

    cms = multilabel_confusion_matrix(dataset.labels, preds)
    cr = classification_report(dataset.labels, preds)
    
    importance_matrix = torch.cat(importance_list).to(device)
    wid_frame_matrix = torch.cat(frame_wid_list).to(device)
    wid_obj_matrix = torch.cat(obj_wid_list).to(device)
    frame_spearman = spearman_correlation(wid_frame_matrix, importance_matrix)
    obj_spearman = spearman_correlation(wid_obj_matrix, importance_matrix)

    return map_micro, map_macro, acc, frame_spearman, obj_spearman, cms, cr

def main():
    if args.dataset == 'cufed':
        dataset = CUFED(root_dir=args.dataset_root, feats_dir=args.feats_dir, split_dir=args.split_dir, is_train=False)
    else:
        sys.exit("Unknown dataset!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    model = Model(args.gcn_layers, dataset.NUM_FEATS, dataset.NUM_CLASS).to(device)
    data = torch.load(args.model[0])
    print("load model from epoch {}".format(data['epoch']))
    model.load_state_dict(data['model_state_dict'])

    if args.verbose:
        print("running on {}".format(device))
        print("test_set={}".format(len(dataset)))

    out_file = None
    if args.save_scores:
        out_file = open(args.save_path, 'w')

    t0 = time.perf_counter()
    map_micro, map_macro, acc, spearman_global, spearman_local, cms, cr = evaluate(model, dataset, loader, out_file, device)
    t1 = time.perf_counter()

    if args.save_scores:
        out_file.close()

    print('map_micro={:.2f} map_macro={:.2f} accuracy={:.2f} spearman_global={:.2f} spearman_local={:.2f} dt={:.2f}sec'.format(map_micro, map_macro, acc * 100, spearman_global, spearman_local, t1 - t0))
    print(cr)
    showCM(cms)


if __name__ == '__main__':
    main()