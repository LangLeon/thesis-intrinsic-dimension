import torch


def train_epoch(model, train_loader, val_loader, optimizer, loss_function, device, print_freq, print_prec):
    model.train()
    train_loss, train_acc = epoch_iter(model, train_loader, optimizer, loss_function, device, print_freq, print_prec)
    print("Train Epoch over. train_loss: {}; train_accuracy: {} \n".format(round(train_loss, print_prec), round(train_acc, print_prec)))

    with torch.no_grad():
        model.eval()
        val_loss, val_acc = epoch_iter(model, val_loader, optimizer, loss_function, device, print_freq, print_prec)
        print("Val Epoch over. val_loss: {}; val_accuracy: {} \n".format(val_loss, val_acc))
    return train_loss, train_acc, val_loss, val_acc


def epoch_iter(model, data, optimizer, loss_function, device, print_freq, print_prec):
    t = 0
    total_loss = 0
    total_accuracy = 0

    for (x, label) in data:

        loss, accuracy = train_batch(model, (x, label), optimizer, loss_function, device)
        if t % print_freq == 0:
            print("Batch: {}; loss: {}; acc: {}".format(t, round(loss,print_prec), round(accuracy,print_prec)))
        total_loss += loss
        total_accuracy += accuracy
        t+= 1

    loss_avg = total_loss / t
    accuracy_avg = total_accuracy / t

    return loss_avg, accuracy_avg


def train_batch(model, batch, optimizer, loss_function, device):
    image, label = batch
    image = image.to(device)
    label = label.to(device)
    optimizer.zero_grad()
    prediction = model(image)
    loss = torch.sum(loss_function(prediction, label))
    if model.training:
        loss.backward()
        optimizer.step()
    accuracy = (torch.argmax(prediction, 1) == label).sum().float() / prediction.shape[0]
    return loss.item(), accuracy.item()
