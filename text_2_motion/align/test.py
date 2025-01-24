# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json

from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
import yaml
from box import Box
import torch
from .model import TextKPAlignmentModel
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import math
from os import path as op
normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"
torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    args = train_args()
    fixseed(args.seed)
    config = Box(yaml.load(open(op.join(args.save_dir, 'model.yaml')), Loader=yaml.FullLoader))

    dist_util.setup_dist(args.device)

    print("creating model...")
    model = TextKPAlignmentModel(**config.model).to(config.model.device)
    model.load_state_dict(torch.load(op.join(args.save_dir, 'align_model_200.pth')))
    model.eval()
    file_name = "M001234"
    dataset_dir = "./dataset/HumanML3D"
    kp = torch.tensor(np.load(op.join(dataset_dir, f"kp_joint_vecs/{file_name}.npy")))[0].float().cuda()
    text_embedding = np.load(op.join(dataset_dir, f"npz/{file_name}.npz"))
    text_embedding = [torch.tensor(text_embedding['spt_encoding'][t, :text_embedding['len'][t]]).float().unsqueeze(1).cuda() for t in range(len(text_embedding['spt_encoding']))]
    idx = 0
    # import pdb; pdb.set_trace()
    T = len(kp)
    n = len(text_embedding[idx])
    if n == 1:
        lT = T
        starts = torch.tensor([0])
    else:
        lamb = (1/math.log(n+2)+1)/n
        lT = int(T*lamb)
        step = int((1-lamb)/(n-1)*T)
        starts = torch.arange(0, n*step, step)
    idxx = (starts[:, None] + torch.arange(lT)).tolist()
    kp = [kp[starts[:, None] + torch.arange(lT), :]]
    text_features = model.text_feature_extractor([text_embedding[idx]], len(text_embedding[idx]), [len(text_embedding[idx])])
    decoder_ret = model.domain_model(kp, [text_embedding[idx]], [len(text_embedding[idx])])
    weight = model.get_weight_distribution(decoder_ret, 0, kp[0].shape).cpu().tolist()
    print(weight)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6), dpi=400)
    for w in range(len(weight)):
        plt.plot(
            # idxx[w], 
            weight[w][0])
    plt.savefig(op.join(args.save_dir, 'weight1.png'))
    
    
if __name__ == "__main__":
    main()
