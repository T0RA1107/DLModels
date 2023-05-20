import numpy as np

def train(model, dataloader_train, dataloader_valid, loss_fn, optimizer, config):
    n_epochs = config.train.n_epochs
    device = config.train.device
    train_score = []
    valid_score = []
    train_losses = []
    valid_losses = []
    for epoch in range(1, n_epochs + 1):
        n_train = 0
        acc_train = 0
        losses_train = []

        model.train()
        
        for x, t in dataloader_train:
            n_train += t.size()[0]
            model.zero_grad()

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
        for x, t in dataloader_valid:
            n_valid += t.size()[0]
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
        
        if epoch % 10 == 0:
            print(f"EPOCH: {epoch}")
            print("train loss: {:.3f}  train accuracy: {:.3f}".format(loss_train, accuracy_t))
            print("valid loss: {:.3f}  valid accuracy: {:.3f}".format(loss_valid, accuracy_v))
        
    return train_score, valid_score, train_losses, valid_losses