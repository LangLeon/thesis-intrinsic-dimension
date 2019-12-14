import torch



def train_epoch(model, train_loader, val_loader, optimizer, loss_function, device, E, params_d, params_0):
    model.train()
    train_loss, train_acc = epoch_iter(model, train_loader, optimizer, loss_function, device, E, params_d, params_0)
    print("Train Epoch over. train_loss: {}; train_accuracy: {} \n".format(train_loss, train_acc))

    model.eval()
    val_loss, val_acc = epoch_iter(model, val_loader, optimizer, loss_function, device, E, params_d, params_0)
    print("Val Epoch over. val_loss: {}; val_accuracy: {} \n".format(val_loss, val_acc))
    return train_loss, train_acc, val_loss, val_acc


def epoch_iter(model, data, optimizer, loss_function, device, E, params_d, params_0):
    t = 0
    total_loss = 0
    total_accuracy = 0

    for (x, label) in data:

        loss, accuracy = train_batch(model, (x, label), optimizer, loss_function, device, E, params_d, params_0)
        if t % 20 == 0:
            print("Batch: {}; loss: {}; acc: {}".format(t, round(loss,2), round(accuracy,2)))
        total_loss += loss
        total_accuracy += accuracy
        t+= 1

    loss_avg = total_loss / t
    accuracy_avg = total_accuracy / t

    return loss_avg, accuracy_avg


def train_batch(model, batch, optimizer, loss_function, device, E, params_d, params_0):
    image, label = batch
    image.to(device)
    label.to(device)
    optimizer.zero_grad()
    prediction = model(image)
    loss = torch.sum(loss_function(prediction, label))
    if model.training:
        loss.backward()
        # optimizer.step() Don't just do an optimizer step, since we will do it manually anyway right now!
    accuracy = (torch.argmax(prediction, 1) == label).sum().float() / prediction.shape[0]
    return loss.item(), accuracy.item()
