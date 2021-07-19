import os
import time
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm

from sklearn import svm
from sklearn.metrics import classification_report

from hats_pytorch import HATS
from demo.dataset import PseeDataset, CenterTransform, collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    return args


def predict(hats, loader, name=""):

    t_tot = 0
    histograms = []
    classes = []
    with torch.no_grad():
        for events, lengths, labels in tqdm(loader):
            events = events.to('cuda')
            lengths = lengths.to('cuda')
            t_rec = time.time()
            hists = hats(events, lengths)
            t_tot += time.time() - t_rec

            hists = hists.cpu().numpy()
            histograms.append(hists)
            classes.append(labels)

    print("Mean {} HATS computation time: {:.2f} ms / sample".format(
        name, 1000 * t_tot / len(loader.dataset)))

    histograms = np.concatenate(histograms, axis=0)
    histograms = histograms.reshape(histograms.shape[0], -1)
    classes = np.concatenate(classes, axis=0)

    return histograms, classes


def main():
    args = parse_args()

    transform = CenterTransform((100, 120))
    hats = HATS((100, 120), r=3, k=10, tau=1e9, delta_t=100e3, bins=1, fold=False)
    hats.to('cuda')
    hats.eval()

    train_dataset = PseeDataset(os.path.join(args.data_dir, "train"), transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               shuffle=True,
                                               batch_size=args.batch_size,
                                               collate_fn=collate_fn)
    test_dataset = PseeDataset(os.path.join(args.data_dir, "test"), transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              collate_fn=collate_fn)

    np.random.seed(42)
    lsvm = svm.LinearSVC(max_iter=100000, C=1)

    train_hist, train_labels = predict(hats, train_loader, "train")
    lsvm.fit(train_hist, train_labels)

    test_hist, test_labels = predict(hats, test_loader, "test")
    pred_labels = lsvm.predict(test_hist)

    report = classification_report(test_labels, pred_labels,
                                   target_names=train_dataset.names)
    print(report)


if __name__ == "__main__":
    main()
