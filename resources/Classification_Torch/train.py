import sys
import time
import torch
import numpy as np


def save_checkpoint (state, filename):
    """ saving model's weights """
    print ('=> saving checkpoint')
    torch.save (state, filename)


def load_checkpoint (checkpoint, model):
    """ loading model's weights """
    print ('=> loading checkpoint')
    model.load_state_dict (checkpoint ['state_dict'])


def accuracy (outputs, labels):
    """ calculate percent of true labels """
    # predicted labels
    _, preds = torch.max (outputs, dim = 1)
    return torch.sum (preds == labels).item () / len (preds)


def train (model, loader, loss_fn, optimizer, metric_fn):
    """ training one epoch and calculate loss and metrics """
    
    # Training model
    model.train ()
    losses = 0.0
    metrics = 0.0
    steps = len (loader)

    for i, (inputs, labels) in enumerate (loader):
        # Place to gpu
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model (inputs)

        # Calculate loss
        loss = loss_fn (outputs, labels)
        losses += loss

        # Backpropagation and update weights
        optimizer.zero_grad ()
        loss.backward ()
        optimizer.step ()


        # Calculate metrics
        metric = metric_fn (outputs, labels)
        metrics += metric

        # report
        sys.stdout.flush ()
        sys.stdout.write ('\r Step: [%2d/%2d], loss: %.4f - acc: %.4f' % (i, steps, loss.item (), metric))
    sys.stdout.write ('\r')
    return losses.item () / len (loader), metrics / len (loader)


def evaluate (model, loader, loss_fn, metric_fn):
    """ Evaluate trained weights using calculate loss and metrics """
    # Evaluate model
    model.eval ()
    losses = 0.0
    metrics = 0.0

    with torch.no_grad ():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model (inputs)
            loss = loss_fn (outputs, labels)
            losses += loss

            metrics += metric_fn (outputs, labels)

    return losses.item () / len (loader), metrics / len (loader)


def fit (model, train_dl, valid_dl, loss_fn, optimizer, num_epochs, metric_fn, checkpoint_path, scheduler = None, load_model = False):
    """ fiting model to dataloaders, saving best weights and showing results """
    losses, val_losses, accs, val_accs = [], [], [], []
    best_acc = 0.0

    # to continue training from saved weights
    if load_model:
        load_checkpoint (torch.load (checkpoint_path), model)

    since = time.time ()

    for epoch in range (num_epochs):

        loss, acc = train (model, train_dl, loss_fn, optimizer, metric_fn)
        val_loss, val_acc = evaluate (model, valid_dl, loss_fn, metric_fn)

        
        losses.append (loss)
        accs.append (acc)
        val_losses.append (val_loss)
        val_accs.append (val_acc)
        
        # learning rate scheduler
        if scheduler is not None:
            scheduler.step (val_acc)

        # save weights if improved
        if val_acc > best_acc:
            checkpoint = {'state_dict': model.state_dict (), 'optimizer': optimizer.state_dict ()}
            save_checkpoint (checkpoint, checkpoint_path)
            best_acc = val_acc

        print ('Epoch [{}/{}], loss: {:.4f} - acc: {:.4f} - val_loss: {:.4f} - val_acc: {:.4f}'.format (epoch + 1, num_epochs, loss, acc, val_loss, val_acc))

    period = time.time () - since
    print ('Training complete in {:.0f}m {:.0f}s'.format (period // 60, period % 60))

    return dict (loss = losses, val_loss = val_losses, acc = accs, val_acc = val_accs)