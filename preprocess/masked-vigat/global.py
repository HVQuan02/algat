import torch
import numpy as np
import sys
import os
import json
from PIL import Image
from transformers import AutoImageProcessor, ViTModel, CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader, Dataset
import argparse

parser = argparse.ArgumentParser(description='ViGAT global feature processing')
parser.add_argument('--extractor', type=str, default='clip', choices=['clip', 'vit'], help='global feature extractor')
parser.add_argument('--preprocess_dir', type=str, default='/kaggle/input/cufed-full-split', help='preprocess directory')
parser.add_argument('--save_dir', type=str, default='/kaggle/working/global_feat', help='save directory for global feature')
parser.add_argument('--num_workers', type=int, default=2, help='num of workers of data loader')
parser.add_argument('--sample_size', type=int, default=30, help='sampling number of images in an album')
args = parser.parse_args()

event_labels = ['Architecture', 'BeachTrip', 'Birthday', 'BusinessActivity',
                  'CasualFamilyGather', 'Christmas', 'Cruise', 'Graduation',
                  'GroupActivity', 'Halloween', 'Museum', 'NatureTrip',
                  'PersonalArtActivity', 'PersonalMusicActivity', 'PersonalSports',
                  'Protest', 'ReligiousActivity', 'Show', 'Sports', 'ThemePark',
                  'UrbanTrip', 'Wedding', 'Zoo']

class AlbumDataset(Dataset):
  def __init__(self, album_names):
    self.album_names = album_names

  def __len__(self):
    return len(self.album_names)

  def __getitem__(self, idx):
    return self.album_names[idx]

def my_collate(batch):
    data = [item for item in batch]
    return data

def global_masked(args):
  datasets_root = "/kaggle/input/thesis-cufed"
  dataset_root = os.path.join(datasets_root, 'CUFED')
  label_path = os.path.join(dataset_root, 'event_type.json')
  dataset_path = os.path.join(dataset_root, 'images')
  album_imgs_path = '/kaggle/working/preprocess/album_imgs.json'

  with open(label_path, 'r') as f:
    album_types = json.load(f)

  preprocess_path = os.path.join(args.preprocess_dir, 'full_split.txt')

  if args.extractor == 'vit':
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
  elif args.extractor == 'clip':
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
  else:
    sys.exit("Unknown extractor!")

  album_batch_size = 1

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  with open(preprocess_path, 'r') as f:
    album_names = [name.strip() for name in f]

  album_dataset = AlbumDataset(album_names)
  album_loader = DataLoader(album_dataset, batch_size=album_batch_size, num_workers=args.num_workers, collate_fn=my_collate)

  with open(album_imgs_path, 'r') as json_file:
    album_imgs_dict = json.load(json_file)

  for album_idx, albums in enumerate(album_loader):
    album = albums[0]
    
    # skip pre-existing npy feats
    global_path = os.path.join(args.save_dir, f"{album}.npy")
    if os.path.exists(global_path):
      continue

    print(f"------Album {album_idx + 1}: {album}------")

    album_dir = os.path.join(dataset_path, album)
    image_names = os.listdir(album_dir)
    image_paths = [os.path.join(album_dir, image_name) for image_name in image_names]

    # vit_global_processor
    with torch.no_grad():
      inputs = processor(text=album_types[album], images=[Image.open(image_path) for image_path in image_paths], 
                             return_tensors="pt", padding=True)
      outputs = model(**inputs.to(device))
    sims = outputs.logits_per_image.cpu().numpy().mean(axis=1)
    feat = outputs.image_embeds.cpu().numpy()
    selected_feat = np.zeros((args.sample_size, feat.shape[1]), dtype=np.float32)
    top_sims = np.argsort(sims)[-args.sample_size:][::-1]

    for i, idx in enumerate(top_sims):
      selected_feat[i] = feat[idx]
    np.save(os.path.join(args.save_dir, f"{album}.npy"), selected_feat)

    # get selected images of albums
    if album in album_imgs_dict:
      continue
    image_names_without_extension = np.char.replace(image_names, '.jpg', '')
    album_imgs_dict[album] = image_names_without_extension[top_sims]
    with open(album_imgs_path, 'w') as json_file:
      json.dump(album_imgs_dict, json_file)

def main():
  global_masked(args)


if __name__ == '__main__':
  main()