import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import json

class ConceptualDataset(data.Dataset):
    def __init__(self, opt):
        self.path = os.path.join(opt.dataroot, 'images')
        if opt.isTrain:
            self.ids = json.load(open(os.path.join(opt.dataroot, 'val_index.json'), 'r'))
        else:
            self.ids = json.load(open(os.path.join(opt.dataroot, 'val_index.json'), 'r'))

        transforms_list = []
        transforms_list.append(transforms.Resize((opt.img_size, opt.img_size)))
        transforms_list += [transforms.ToTensor()]
        transforms_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transforms_list)

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        img_id = self.ids[index]
        image = Image.open(os.path.join(self.path, img_id)).convert('RGB')
        image = self.transform(image)

        return image

    def __len__(self):
        return len(self.ids)
