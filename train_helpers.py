import torch


def train_epoch(model, train_loader, val_loader, optimizer, loss_function, ARGS):
    model.train()
    train_loss, train_acc = epoch_iter(model, train_loader, optimizer, loss_function, ARGS)
    print("Train Epoch over. train_loss: {}; train_accuracy: {} \n".format(round(train_loss, ARGS.print_prec), round(train_acc, ARGS.print_prec)))

    if ARGS.parameter_correction:
        optimizer.parameter_correction() # Makes sure that the parameters are projected back into the subspace if there was a "drift" along the way

    with torch.no_grad():
        model.eval()
        val_loss, val_acc = epoch_iter(model, val_loader, optimizer, loss_function, ARGS)
        print("Val Epoch over. val_loss: {}; val_accuracy: {} \n".format(val_loss, val_acc))

    subspace_distance = None
    if ARGS.subspace_training:
        subspace_distance = optimizer.compute_subspace_distance()

    return train_loss, train_acc, val_loss, val_acc, subspace_distance


def epoch_iter(model, data, optimizer, loss_function, ARGS):
    t = 0
    total_loss = 0
    total_accuracy = 0

    for (x, label) in data:

        loss, accuracy = train_batch(model, (x, label), optimizer, loss_function, ARGS)
        if t % ARGS.print_freq == 0:
            print("Batch: {}; loss: {}; acc: {}".format(t, round(loss,ARGS.print_prec), round(accuracy,ARGS.print_prec)))
        total_loss += loss
        total_accuracy += accuracy
        t+= 1

    loss_avg = total_loss / t
    accuracy_avg = total_accuracy / t

    return loss_avg, accuracy_avg


def train_batch(model, batch, optimizer, loss_function, ARGS):
    image, label = batch
    image = image.to(ARGS.device)
    label = label.to(ARGS.device)
    optimizer.zero_grad()
    prediction = model(image)
    loss = torch.sum(loss_function(prediction, label))
    if model.training:
        loss.backward()
        optimizer.step()
    accuracy = (torch.argmax(prediction, 1) == label).sum().float() / prediction.shape[0]
    return loss.item(), accuracy.item()
