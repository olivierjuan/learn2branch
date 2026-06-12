import os
import importlib
import argparse
import sys
import pathlib
import pickle
import numpy as np
from time import strftime
from shutil import copyfile
import gzip

import torch
from torch.utils.data import Dataset, DataLoader

import utilities
from utilities import log

from utilities_tf import load_batch_gcnn


class GCNNDataset(Dataset):
    def __init__(self, sample_files):
        self.sample_files = sample_files

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        return self.sample_files[idx]


def collate_fn(batch):
    return load_batch_gcnn(batch)


def pretrain(model, dataloader, device):
    """
    Pre-normalizes a model (i.e., PreNormLayer layers) over the given samples.

    Parameters
    ----------
    model : model.BaseModel
        A base model, which may contain some model.PreNormLayer layers.
    dataloader : DataLoader
        Dataset loader to use for pre-training the model.
    Return
    ------
    number of PreNormLayer layers processed.
    """
    model.pre_train_init()
    i = 0
    while True:
        with torch.no_grad():
            for batch in dataloader:
                c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores = [b.to(device) for b in batch]
                batched_states = (c, ei, ev, v, n_cs, n_vs)

                if not model.pre_train(batched_states):
                    break

        res = model.pre_train_next()
        if res is None:
            break
        else:
            layer, name = res

        i += 1

    return i


def process(model, dataloader, top_k, device, optimizer=None):
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))

    n_samples_processed = 0
    for batch in dataloader:
        c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores = [b.to(device) for b in batch]
        batched_states = (c, ei, ev, v, torch.sum(n_cs, dim=0, keepdim=True), torch.sum(n_vs, dim=0, keepdim=True))  # prevent padding
        batch_size = len(n_cs)

        if optimizer:
            optimizer.zero_grad()
            logits = model(batched_states, training=True) # training mode
            logits = torch.squeeze(logits, 0)[cands.long()].unsqueeze(0)  # filter candidate variables
            logits = model.pad_output(logits, n_cands)  # apply padding now
            loss = torch.nn.functional.cross_entropy(logits, best_cands.long())
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(batched_states, training=False)  # eval mode
                logits = torch.squeeze(logits, 0)[cands.long()].unsqueeze(0)  # filter candidate variables
                logits = model.pad_output(logits, n_cands)  # apply padding now
                loss = torch.nn.functional.cross_entropy(logits, best_cands.long())

        with torch.no_grad():
            true_scores = model.pad_output(cand_scores.reshape(1, -1), n_cands)
            true_bestscore = torch.max(true_scores, dim=-1, keepdim=True)[0]
            true_scores = true_scores.cpu().numpy()
            true_bestscore = true_bestscore.cpu().numpy()

        kacc = []
        for k in top_k:
            pred_top_k = torch.topk(logits, k=k, dim=-1)[1].cpu().numpy()
            pred_top_k_true_scores = np.take_along_axis(true_scores, pred_top_k, axis=1)
            kacc.append(np.mean(np.any(pred_top_k_true_scores == true_bestscore, axis=1)))
        kacc = np.asarray(kacc)

        mean_loss += loss.item() * batch_size
        mean_kacc += kacc * batch_size
        n_samples_processed += batch_size

    mean_loss /= n_samples_processed
    mean_kacc /= n_samples_processed

    return mean_loss, mean_kacc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-m', '--model',
        help='GCNN model to be trained.',
        type=str,
        default='baseline',
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=0,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    ### HYPER PARAMETERS ###
    max_epochs = 1000
    epoch_size = 312
    batch_size = 32
    pretrain_batch_size = 128
    valid_batch_size = 128
    lr = 0.001
    patience = 10
    early_stopping = 20
    top_k = [1, 3, 5, 10]
    train_ncands_limit = np.inf
    valid_ncands_limit = np.inf

    problem_folders = {
        'setcover': 'setcover/500r_1000c_0.05d',
        'cauctions': 'cauctions/100_500',
        'facilities': 'facilities/100_100_5',
        'indset': 'indset/500_4',
    }
    problem_folder = problem_folders[args.problem]

    running_dir = f"trained_models/{args.problem}/{args.model}_torch/{args.seed}"

    os.makedirs(running_dir, exist_ok=True)

    ### LOG ###
    logfile = os.path.join(running_dir, 'log.txt')

    log(f"max_epochs: {max_epochs}", logfile)
    log(f"epoch_size: {epoch_size}", logfile)
    log(f"batch_size: {batch_size}", logfile)
    log(f"pretrain_batch_size: {pretrain_batch_size}", logfile)
    log(f"valid_batch_size : {valid_batch_size }", logfile)
    log(f"lr: {lr}", logfile)
    log(f"patience : {patience }", logfile)
    log(f"early_stopping : {early_stopping }", logfile)
    log(f"top_k: {top_k}", logfile)
    log(f"problem: {args.problem}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed {args.seed}", logfile)

    ### PYTORCH SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(rng.randint(np.iinfo(int).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng.randint(np.iinfo(int).max))

    ### SET-UP DATASET ###
    train_files = list(pathlib.Path(f'data/samples/{problem_folder}/train').glob('sample_*.pkl'))
    valid_files = list(pathlib.Path(f'data/samples/{problem_folder}/valid').glob('sample_*.pkl'))


    def take_subset(sample_files, cands_limit):
        nsamples = 0
        ncands = 0
        for filename in sample_files:
            with gzip.open(filename, 'rb') as file:
                sample = pickle.load(file)

            _, _, _, cands, _ = sample['data']
            ncands += len(cands)
            nsamples += 1

            if ncands >= cands_limit:
                log(f"  dataset size limit reached ({cands_limit} candidate variables)", logfile)
                break

        return sample_files[:nsamples]


    if train_ncands_limit < np.inf:
        train_files = take_subset(rng.permutation(train_files), train_ncands_limit)
    log(f"{len(train_files)} training samples", logfile)
    if valid_ncands_limit < np.inf:
        valid_files = take_subset(valid_files, valid_ncands_limit)
    log(f"{len(valid_files)} validation samples", logfile)

    train_files = [str(x) for x in train_files]
    valid_files = [str(x) for x in valid_files]

    valid_dataset = GCNNDataset(valid_files)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, collate_fn=collate_fn)

    pretrain_files = [f for i, f in enumerate(train_files) if i % 10 == 0]
    pretrain_dataset = GCNNDataset(pretrain_files)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=pretrain_batch_size, shuffle=False, collate_fn=collate_fn)

    ### MODEL LOADING ###
    sys.path.insert(0, os.path.abspath(f'models/{args.model}'))
    import model
    importlib.reload(model)
    model = model.GCNPolicy()
    model.to(device)
    del sys.path[0]

    ### TRAINING LOOP ###
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = np.inf
    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)

        # TRAIN
        if epoch == 0:
            n = pretrain(model=model, dataloader=pretrain_loader, device=device)
            log(f"PRETRAINED {n} LAYERS", logfile)
        else:
            epoch_train_files = rng.choice(train_files, epoch_size * batch_size, replace=True)
            train_dataset = GCNNDataset(epoch_train_files)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            train_loss, train_kacc = process(model, train_loader, top_k, device, optimizer)
            log(f"TRAIN LOSS: {train_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, train_kacc)]), logfile)

        # TEST
        valid_loss, valid_kacc = process(model, valid_loader, top_k, device, None)
        log(f"VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)

        if valid_loss < best_loss:
            plateau_count = 0
            best_loss = valid_loss
            model.save_state(os.path.join(running_dir, 'best_params.pkl'))
            log(f"  best model so far", logfile)
        else:
            plateau_count += 1
            if plateau_count % early_stopping == 0:
                log(f"  {plateau_count} epochs without improvement, early stopping", logfile)
                break
            if plateau_count % patience == 0:
                lr *= 0.2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                log(f"  {plateau_count} epochs without improvement, decreasing learning rate to {lr}", logfile)

    model.restore_state(os.path.join(running_dir, 'best_params.pkl'))
    valid_loss, valid_kacc = process(model, valid_loader, top_k, device, None)
    log(f"BEST VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)
