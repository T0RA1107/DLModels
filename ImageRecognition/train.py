import numpy as np
import torch
import wandb
import os
from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, dynamic_ncols=True)

def train(model, dataloader_train, dataloader_valid, loss_fn, optimizer, config):
    wandb.init(project=config.project, name="train and save weight")
    n_epochs = config.train.n_epochs
    wandb.config.n_epochs = n_epochs
    device = torch.device(config.train.device)
    model.to(device)
    optimizer = optimizer(model.parameters(), lr=config.train.lr)
    train_score = []
    valid_score = []
    train_score_best = 0.0
    valid_score_best = 0.0
    train_losses = []
    valid_losses = []
    for epoch in tqdm(range(1, n_epochs + 1), desc="EPOCHS"):
        n_train = 0
        acc_train = 0
        losses_train = []

        model.train()
        
        for x, t in tqdm(dataloader_train, leave=False, desc="train"):
            n_train += t.size()[0]
            model.zero_grad()

            x = x.permute(0, 2, 3, 1)
            x = x.to(device)
            t = t.to(device)

            y = model(x)

            loss = loss_fn(y, t)

            loss.backward()

            optimizer.step()
        
            pred = y.argmax(1)
            acc_train += (pred == t).float().sum().item()
            losses_train.append(loss.tolist())

        n_valid = 0
        acc_valid = 0
        losses_valid = []

        model.eval()
        for x, t in tqdm(dataloader_valid, leave=False, desc="valid"):
            n_valid += t.size()[0]
            x = x.permute(0, 2, 3, 1)
            x = x.to(device)
            t = t.to(device)

            y = model(x)

            loss = loss_fn(y, t)

            pred = y.argmax(1)
            acc_valid += (pred == t).float().sum().item()
            losses_valid.append(loss.tolist())
        
        accuracy_t = acc_train / n_train
        accuracy_v = acc_valid / n_valid
        train_score.append(accuracy_t)
        valid_score.append(accuracy_v)

        loss_train = np.mean(losses_train)
        loss_valid = np.mean(losses_valid)
        train_losses.append(loss_train)
        valid_losses.append(loss_valid)
        wandb.log({
            'train acc': accuracy_t,
            'valid acc': accuracy_v,
            'train loss': loss_train,
            'valid loss': loss_valid
        })
        
        if accuracy_t > train_score_best and accuracy_v > valid_score_best:
            torch.save(model.state_dict(), os.path.join(config.train.weight_path, 'model_weight.pth'))
            wandb.save(os.path.join(config.train.weight_path, 'model_weight.pth'))
        
        train_score_best = max(train_score_best, accuracy_t)
        valid_score_best = max(valid_score_best, accuracy_v)
    
    wandb.finish()