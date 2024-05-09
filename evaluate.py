import argparse
import time
import torch
import sys
from torch.utils.data import DataLoader

from datasets import CUFED
from utils import AP_partial, spearman_correlation
from model import ModelGCNConcAfter as Model

parser = argparse.ArgumentParser(description='GCN Album Classification')
parser.add_argument('model', nargs=1, help='trained model')
parser.add_argument('--gcn_layers', type=int, default=2, help='number of gcn layers')
parser.add_argument('--dataset', default='cufed', choices=['holidays', 'pec', 'cufed'])
parser.add_argument('--dataset_root', default='/kaggle/input/thesis-cufed/CUFED', help='dataset root directory')
parser.add_argument('--feats_dir', default='/kaggle/input/cufed-feats', help='global and local features directory')
parser.add_argument('--split_dir', default='/kaggle/input/cufed-full-split', help='train split and val split')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_objects', type=int, default=50, help='number of objects with best DoC')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for data loader')
parser.add_argument('--ext_method', default='VIT', choices=['VIT', 'RESNET'], help='Extraction method for features')
parser.add_argument('--save_scores', action='store_true', help='save the output scores')
parser.add_argument('--save_path', default='scores.txt', help='output path')
parser.add_argument('-v', '--verbose', action='store_true', help='show details')
args = parser.parse_args()

def evaluate(model, dataset, loader, out_file, device):
    scores = torch.zeros((len(dataset), dataset.NUM_CLASS), dtype=torch.float32)
    gidx = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            feats, feat_global, _, _ = batch

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
    # Change tensors to 1d-arrays
    scores = scores.numpy()
    map = AP_partial(dataset.labels, scores)[1]
    return map

def main():
    if args.dataset == 'cufed':
        dataset = CUFED(root_dir=args.dataset_root, feats_dir=args.feats_dir, split_dir=args.split_dir, is_train=False, ext_method=args.ext_method)
    else:
        sys.exit("Unknown dataset!")

    device = torch.device('cuda:0')
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    if args.verbose:
        print("running on {}".format(device))
        print("num samples={}".format(len(dataset)))
        print("missing videos={}".format(dataset.num_missing))

    model = Model(args.gcn_layers, dataset.NUM_FEATS, dataset.NUM_CLASS).to(device)
    data = torch.load(args.model[0])
    model.load_state_dict(data['model_state_dict'])

    out_file = None
    if args.save_scores:
        out_file = open(args.save_path, 'w')

    t0 = time.perf_counter()
    map = evaluate(model, dataset, loader, out_file, device)
    t1 = time.perf_counter()

    if args.save_scores:
        out_file.close()

    print('map={:.2f} dt={:.2f}sec'.format(map, t1 - t0))

if __name__ == '__main__':
    main()