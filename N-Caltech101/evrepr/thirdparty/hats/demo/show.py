import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from hats_pytorch import HATS
from demo.dataset import PseeDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    hats = HATS((100, 120), r=1, k=3, tau=1e9, delta_t=100000, fold=True)
    hats.to('cuda')
    hats.eval()

    train_dataset = PseeDataset(os.path.join(args.data_dir, "train"))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               shuffle=True,
                                               batch_size=1,
                                               collate_fn=collate_fn)

    with torch.no_grad():
        for i, (events, lengths, labels) in enumerate(train_loader):
            if i == args.num_samples:
                return

            events = events.to('cuda')
            lengths = lengths.to('cuda')
            hists = hats(events, lengths)
            hists = hists[0].cpu().numpy()

            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(hists[0], cmap='hot')
            ax2.imshow(hists[1], cmap='hot')
            plt.show()


if __name__ == "__main__":
    main()
