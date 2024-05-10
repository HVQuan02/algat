import os
import json
import sys
import numpy as np
from torch.utils.data import Dataset

def get_album_importance(album_imgs, album_importance):
    img_score_dict = {}
    for _, image, score in album_importance:
        img_score_dict[image] = score
    importances = np.zeros(len(album_imgs))
    for i, image in enumerate(album_imgs):
        importances[i] = img_score_dict[image]
    return importances

class CUFED(Dataset):
    NUM_CLASS = 23
    NUM_FRAMES = 30
    NUM_BOXES = 50
    event_labels = ['Architecture', 'BeachTrip', 'Birthday', 'BusinessActivity',
                    'CasualFamilyGather', 'Christmas', 'Cruise', 'Graduation',
                    'GroupActivity', 'Halloween', 'Museum', 'NatureTrip',
                    'PersonalArtActivity', 'PersonalMusicActivity', 'PersonalSports',
                    'Protest', 'ReligiousActivity', 'Show', 'Sports', 'ThemePark',
                    'UrbanTrip', 'Wedding', 'Zoo']

    def __init__(self, root_dir, feats_dir, split_dir, is_train, is_val=False, ext_method= 'VIT'):
        self.root_dir = root_dir
        self.feats_dir = feats_dir
        
        if is_train:
            if is_val:
                self.phase = 'val'
            else:
                self.phase = 'train'
        else:
            self.phase = 'test'
            
        if ext_method == 'VIT':
            self.local_folder = 'vit_local'
            self.global_folder = 'vit_global'
            self.NUM_FEATS = 768
        elif ext_method == 'RESNET':
            self.local_folder = 'R152_local'
            self.global_folder = 'R152_global'
            self.NUM_FEATS = 2048
        else:
            sys.exit("Unknown Extractor")

        if self.phase == 'train':
            split_path = os.path.join(split_dir, 'train_split.txt')
        else:
            split_path = os.path.join(split_dir, 'val_split.txt')

        vidname_list = []
        label_path = os.path.join(root_dir, "event_type.json")
        with open(label_path, 'r') as f:
          album_data = json.load(f)

        if self.phase == 'test':
            importance_path = os.path.join(root_dir, "image_importance.json")
            with open(importance_path, 'r') as f:
                album_importance = json.load(f)

            album_imgs_path = os.path.join(split_dir, "album_imgs.json")
            with open(album_imgs_path, 'r') as f:
                album_imgs = json.load(f)
                
            self.importance = album_importance
            self.album_imgs = album_imgs

        with open(split_path, 'r') as f:
            album_names = f.readlines()
        vidname_list = [name.strip() for name in album_names]

        length = len(vidname_list)
        labels_np = np.zeros((length, self.NUM_CLASS), dtype=np.float32)
        for i, vidname in enumerate(vidname_list):
            for lbl in album_data[vidname]:
                idx = self.event_labels.index(lbl)
                labels_np[i, idx] = 1

        self.labels = labels_np
        self.videos = vidname_list

        self.num_missing = 0  # no missing videos by default!!

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        name = self.videos[idx]
        feats_path = os.path.join(self.feats_dir, self.local_folder, name + '.npy')
        global_path = os.path.join(self.feats_dir, self.global_folder, name + '.npy')
        feats = np.load(feats_path)
        feat_global = np.load(global_path)
        label = self.labels[idx, :]
        if self.phase == 'test':
            album_importance = self.importance[name]
            album_imgs = self.album_imgs[name]
            importances = get_album_importance(album_imgs, album_importance)
            return feats, feat_global, label, importances
        return feats, feat_global, label