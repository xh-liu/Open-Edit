import os
from collections import OrderedDict

import data
from options.train_options import TrainOptions
from models.OpenEdit_model import OpenEditModel
from trainers.OpenEdit_optimizer import OpenEditOptimizer
from util.visualizer import Visualizer
from util import html
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import json
import pdb
import pickle
from util.vocab import Vocabulary

TrainOptions = TrainOptions()
opt = TrainOptions.parse()
opt.gpu = 0

ori_cap = opt.ori_cap.split()
new_cap = opt.new_cap.split()
global_edit = False

alpha = 5
optimize_iter = 10 

opt.world_size = 1
opt.rank = 0
opt.mpdist = False
opt.num_gpu = 1
opt.batchSize = 1
opt.manipulation = True
opt.perturbation = True

open_edit_optimizer = OpenEditOptimizer(opt)
open_edit_optimizer.open_edit_model.netG.eval()

# optimizer
visualizer = Visualizer(opt, rank=0)

# create a webpage that summarizes the all results
web_dir = os.path.join('visual_results', opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# image loader
transforms_list = []
transforms_list.append(transforms.Resize((opt.img_size, opt.img_size)))
transforms_list += [transforms.ToTensor()]
transforms_list += [transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
transform = transforms.Compose(transforms_list)
image = Image.open(opt.img_path).convert('RGB')
image = transform(image)
image = image.unsqueeze(0).cuda()

# text loader
vocab = pickle.load(open('vocab/'+opt.dataset_mode+'_vocab.pkl', 'rb'))
ori_txt = []
ori_txt.append(vocab('<start>'))
for word in ori_cap:
    ori_txt.append(vocab(word))
ori_txt.append(vocab('<end>'))
ori_txt = torch.LongTensor(ori_txt).unsqueeze(0).cuda()
new_txt = []
new_txt.append(vocab('<start>'))
for word in new_cap:
    new_txt.append(vocab(word))
new_txt.append(vocab('<end>'))
new_txt = torch.LongTensor(new_txt).unsqueeze(0).cuda()

data = {'image': image, 'caption': new_txt, 'length': [4]}

# save input image
visuals = OrderedDict([('input_image', image[0])])

# reconstruct original image
reconstructed = open_edit_optimizer.open_edit_model(data, mode='inference')[0]
visuals['reconstructed'] = reconstructed

# manipulate without optimizing perturbations
manipulated_ori = open_edit_optimizer.open_edit_model(data, mode='manipulate', ori_cap=ori_txt, new_cap=new_txt, alpha=alpha)
visuals['manipulated_ori'] = manipulated_ori[0]

# optimize perturbations
for iter_cnt in range(optimize_iter):
    open_edit_optimizer.run_opt_one_step(data, ori_txt, new_txt, alpha, global_edit=global_edit)
    message = '(optimization, iters: %d) ' % iter_cnt
    errors = open_edit_optimizer.get_latest_losses()
    for k, v in errors.items():
        v = v.mean().float()
        message += '%s: %.3f ' % (k, v)
    print(message)

# manipulation results after optimizing perturbations
manipulated_opt = open_edit_optimizer.open_edit_model(data, mode='manipulate', ori_cap=ori_txt, new_cap=new_txt, alpha=alpha)
visuals['optimized_manipulated'] = manipulated_opt[0]


visualizer.save_images(webpage, visuals, [opt.img_path], gray=True)
webpage.save()
