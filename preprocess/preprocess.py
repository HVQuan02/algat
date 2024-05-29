# preprocess black box
vit_local_path = os.path.join(preprocess_dir, 'vit_local_bb')

def vit_globcal_processor():
  album_batch_size = 1
  sample_size = 30
  objects_size = 50
  num_workers = 2
  od_batch_size = 10
  crop_batch_size = 100

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ext_model.to(device)
  od_model.to(device)

  with open(preprocess_path, 'r') as f:
    album_names = [name.strip() for name in f]

  album_dataset = AlbumDataset(album_names)
  album_loader = DataLoader(album_dataset, batch_size=album_batch_size, num_workers=num_workers, collate_fn=my_collate)

  for album_idx, albums in enumerate(album_loader):
    album = albums[0]
    
    # skip pre-existing npy feats
    album_feats_path = os.path.join(vit_local_path, f"{album}.npy")
    if os.path.exists(album_feats_path):
      continue

    print(f"------Album {album_idx + 1}: {album}------")

    album_dir = os.path.join(dataset_path, album)
    image_paths = [os.path.join(album_dir, filename) for filename in os.listdir(album_dir)]
    image_dataset = ImageDataset(image_paths, transform=transforms, read_image=read_image)
    image_loader = DataLoader(image_dataset, batch_size=od_batch_size, num_workers=num_workers, collate_fn=my_collate)

    # detect objects and sort
    result_list = []
    gidx = 0
    od_model.eval()
    with torch.no_grad():
      for images in image_loader:
        results = od_model([image.to(device) for image in images])
        for i in range(len(results)):
          results[i]['idx'] = gidx + i
          results[i]['feat'] = images[i]
        gidx += len(results)
        result_list += results
    # sort images by number of detected objects
    sorted_result_list = sorted(result_list, key=lambda d: len(d['boxes']), reverse=True)
    sample_list = sorted_result_list[:sample_size]
    sample_paths = [image_paths[sample['idx']] for sample in sample_list]

    # vit_global_processor
    with torch.no_grad():
      inputs = ext_processor(images=[Image.open(image_path) for image_path in sample_paths], return_tensors="pt", padding=True)
      outputs = ext_model(**inputs.to(device))
      features = outputs.last_hidden_state
      print('feats: ', features.shape)
      album_features = features.cpu().numpy().mean(axis=1).astype(np.float32)
      print('album feats: ', features.shape)
      np.save(os.path.join(vit_global_path, f"{album}.npy"), album_features)

    # crop objects
    cropped_image_tensor_list = []
    pad_list = []
    gidx = 0
    for sample in sample_list:
      boxes = sample['boxes']
      pad_size = objects_size
      loop_cnt = min(len(boxes), objects_size)
      for box_idx in range(loop_cnt):
        x1, y1, x2, y2 = [round(cord) for cord in boxes[box_idx].tolist()]
        cropped_image_tensor = crop(sample['feat'], y1, x1, y2 - y1, x2 - x1)
        cropped_image_tensor = torch.clamp(cropped_image_tensor, min=0.0, max=1.0)
        cropped_image_tensor_list.append(cropped_image_tensor)
      gidx += loop_cnt
      pad_size -= loop_cnt
      pad_list.append((gidx, pad_size))

    # vit_local_processor
    crop_dataset = CropDataset(cropped_image_tensor_list)
    crop_loader = DataLoader(crop_dataset, batch_size=crop_batch_size, num_workers=num_workers, collate_fn=my_collate)
    vit_local_list = []
    with torch.no_grad():
      for crops in crop_loader:
        cropped_inputs = vit_processor(crops, return_tensors="pt", do_rescale=False)
        cropped_outputs = vit_model(**cropped_inputs.to(device))
        cropped_features = cropped_outputs.last_hidden_state
        cropped_features = cropped_features.cpu().numpy().mean(axis=1)
        vit_local_list.append(cropped_features)
      vit_local_np = np.vstack(vit_local_list)
      if len(cropped_image_tensor_list) < sample_size * objects_size:
        for pad in reversed(pad_list):
          vit_local_np = np.vstack([vit_local_np[:pad[0]], np.zeros((pad[1], 768)), vit_local_np[pad[0]:]])
      vit_local_np = vit_local_np.reshape(sample_size, objects_size, -1).astype(np.float32)
      np.save(os.path.join(vit_local_path, f"{album}.npy"), vit_local_np)

# get album images importance
# process_path = preprocess_path # test
process_path = '/kaggle/input/cufed-full-split/full_split.txt'

def get_album_image_format(full_path):
    # Split the path based on '/'
    parts = full_path.split('/')

    # Get the second-to-last part and remove the file extension
    desired_part = parts[-2] + '/' + parts[-1][:-4]
    return desired_part

def get_album_imgs():
  album_batch_size = 1
  sample_size = 30
  num_workers = 2
  fasterrcnn_batch_size = 10
  with open('/kaggle/input/cufed-full-split/album_imgs.json', 'r') as json_file:
    album_images_dict = json.load(json_file)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  fasterrcnn_model.to(device)

  with open(process_path, 'r') as f:
    album_names = [name.strip() for name in f]

  album_dataset = AlbumDataset(album_names)
  album_loader = DataLoader(album_dataset, batch_size=album_batch_size, num_workers=num_workers, collate_fn=my_collate)

  for album_idx, albums in enumerate(album_loader):
    album = albums[0]
    if album in album_images_dict:
        continue

    print(f"------Album {album_idx + 1}: {album}------")

    album_dir = os.path.join(dataset_path, album)
    image_paths = [os.path.join(album_dir, filename) for filename in os.listdir(album_dir)]
    image_dataset = ImageDataset(image_paths, transform=transforms, read_image=read_image)
    image_loader = DataLoader(image_dataset, batch_size=fasterrcnn_batch_size, num_workers=num_workers, collate_fn=my_collate)

    # detect objects and sort
    result_list = []
    gidx = 0
    fasterrcnn_model.eval()
    with torch.no_grad():
      for images in image_loader:
        results = fasterrcnn_model([image.to(device) for image in images])
        for i in range(len(results)):
          results[i]['idx'] = gidx + i
          results[i]['feat'] = images[i]
        gidx += len(results)
        result_list += results
        
    # sort images by number of detected objects
    sorted_result_list = sorted(result_list, key=lambda d: len(d['boxes']), reverse=True)
    sample_list = sorted_result_list[:sample_size]
    sample_paths = [get_album_image_format(image_paths[sample['idx']]) for sample in sample_list]
    album_images_dict[album] = sample_paths
  with open('/kaggle/working/preprocess/album_imgs.json', 'w') as json_file:
    json.dump(album_images_dict, json_file)
    
get_album_imgs()


# preprocess dồn objects
vit_local_path = os.path.join(preprocess_dir, 'vit_local_obj')

def vit_globcal_processor():
  album_batch_size = 1
  sample_size = 30
  objects_size = 50
  num_workers = 2
  fasterrcnn_batch_size = 10
  crop_batch_size = 100

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  vit_model.to(device)
  fasterrcnn_model.to(device)

  with open(preprocess_path, 'r') as f:
    album_names = [name.strip() for name in f]

  album_dataset = AlbumDataset(album_names)
  album_loader = DataLoader(album_dataset, batch_size=album_batch_size, num_workers=num_workers, collate_fn=my_collate)
    
  for album_idx, albums in enumerate(album_loader):
    album = albums[0]
    
    # skip pre-existing npy feats
    album_feats_path = os.path.join(vit_local_path, f"{album}.npy")
    if os.path.exists(album_feats_path):
      continue

    print(f"------Album {album_idx + 1}: {album}------")

    album_dir = os.path.join(dataset_path, album)
    image_paths = [os.path.join(album_dir, filename) for filename in os.listdir(album_dir)]
    image_dataset = ImageDataset(image_paths, transform=transforms, read_image=read_image)
    image_loader = DataLoader(image_dataset, batch_size=fasterrcnn_batch_size, num_workers=num_workers, collate_fn=my_collate)

    # detect objects and sort
    result_list = []
    gidx = 0
    fasterrcnn_model.eval()
    with torch.no_grad():
      for images in image_loader:
        results = fasterrcnn_model([image.to(device) for image in images])
        for i in range(len(results)):
          results[i]['idx'] = gidx + i
          results[i]['feat'] = images[i]
        gidx += len(results)
        result_list += results
    # sort images by number of detected objects
    sorted_result_list = sorted(result_list, key=lambda d: len(d['boxes']), reverse=True)
    sample_list = sorted_result_list[:sample_size]
    sample_paths = [image_paths[sample['idx']] for sample in sample_list]

    # crop objects
    cropped_image_tensor_list = []
    for sample in sorted_result_list:
      boxes = sample['boxes']
      for box in boxes:
        x1, y1, x2, y2 = [round(cord) for cord in box.tolist()]
        cropped_image_tensor = crop(sample['feat'], y1, x1, y2 - y1, x2 - x1)
        cropped_image_tensor = torch.clamp(cropped_image_tensor, min=0.0, max=1.0)
        cropped_image_tensor_list.append(cropped_image_tensor)

    # vit_local_processor
    crop_size = len(cropped_image_tensor_list)
    standard_size = sample_size * objects_size
    remains = standard_size - crop_size
    if remains < 0:
      cropped_image_tensor_list = cropped_image_tensor_list[:standard_size]
    crop_dataset = CropDataset(cropped_image_tensor_list)
    crop_loader = DataLoader(crop_dataset, batch_size=crop_batch_size, num_workers=num_workers, collate_fn=my_collate)
    vit_local_list = []
    with torch.no_grad():
      for crops in crop_loader:
        cropped_inputs = vit_processor(crops, return_tensors="pt", do_rescale=False)
        cropped_outputs = vit_model(**cropped_inputs.to(device))
        cropped_features = cropped_outputs.last_hidden_state
        cropped_features = cropped_features.cpu().numpy().mean(axis=1)
        vit_local_list.append(cropped_features)
      if remains > 0:
        vit_local_list.append(np.zeros((remains, 768)))
      vit_local_np = np.vstack(vit_local_list).reshape(sample_size, objects_size, -1).astype(np.float32)
      np.save(os.path.join(vit_local_path, f"{album}.npy"), vit_local_np)