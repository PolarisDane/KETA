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

normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"
torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    args = train_args()
    fixseed(args.seed)
    config = Box(yaml.load(open(args.config), Loader=yaml.FullLoader))

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
     
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames)

    print("creating model...")
    model = TextKPAlignmentModel(**config.model).to(config.model.device)
    
    epochs = 750
    losses = []
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in tqdm(range(epochs)):
        losses.append([])
        pbar = tqdm(total=len(data), leave=False)
        for motion, cond in data:
            # if epoch == 0:
            #     print(cond['y']['caption'])
            kps = motion[2]
            ori_embedding = cond['y']['ori_encoding']
            spt_embedding = cond['y']['spt_encoding']
            spt_len = cond['y']['spt_len']
            kps_proc = []
            for i in range(len(kps)):
                kp = kps[i]
                T = len(kp)
                n = spt_len[i]
                if n == 1:
                    lT = T
                    starts = torch.tensor([0])
                else:
                    lamb = (1/torch.log(n+2)+1)/n
                    lT = int(T*lamb)
                    step = int((1-lamb)/(n-1)*T)
                    starts = torch.arange(0, n*step, step)
                kps_proc.append(kp[starts[:, None] + torch.arange(lT), :])
                
            # kps_proc = []
            # text_embedding = []
            
            # for i in range(len(spt_len)):
            #     kps_proc.append(kps[i].T.cuda())
            #     text_embedding.append(torch.cat([ori_embedding[i].unsqueeze(0), spt_embedding[i][:, :spt_len[i]].T], dim=0).cuda())
                
            model.train()
            model.zero_grad()
            loss = model(spt_embedding, kps_proc, max(spt_len), spt_len)
            loss.backward()
            optim.step()
            
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)
            losses[-1].append(loss.item())
        pbar.close()
        # save model
        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'align_model_{epoch+1}.pth'))
    
            # vis losses
        
        avg_losses = [np.mean(epoch_losses) for epoch_losses in losses]

        window_length = 5
        polyorder = 2

        plt.figure(figsize=(10, 6), dpi=400)

        plt.plot(range(1, len(avg_losses) + 1), avg_losses, label='Average Loss', color='blue', marker='o', linestyle='-', markersize=6)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Average Loss Curve for Each Epoch with Smoothing')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.save_dir, f'loss_curve.png'))
        plt.close()
if __name__ == "__main__":
    main()
