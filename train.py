""" Training loop using C3D model """

from comet_ml import Experiment
import os.path
import torch
from load_dataset import number_to_lab
import pickle
from torch.utils.data import TensorDataset, DataLoader
import argparse
import numpy as np
from C3D_model import C3D
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import sklearn.metrics
from PIL import Image
import torchvision.transforms as transforms


def plot_confusion_matrix(cm, classes, step, exp, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    exp.log_figure(figure_name=title, figure=plt, step=step)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train C3D")

    parser.add_argument("--epochs", dest="epochs", default=1, help="number of epochs")
    parser.add_argument("--batch_size", dest="batch_size", default=6, help="Batch size")
    parser.add_argument("--lr", dest="lr", default=0.001, help="learning rate train")
    # parser.add_argument("--weight_decay", dest="weight_decay", default=0., help="weight decay")
    # parser.add_argument("--val_perc", dest="val_perc", default=0, help="% validation set")

    # iperparametri C3D

    # train/valid pickle
    parser.add_argument("--train", dest="train", default=None, help="path to train pickle file")
    parser.add_argument("--valid", dest="valid", default=None, help="path to validation pickle file")

    parser.add_argument("--labelling", dest="labelling", default=1, help="1 for user labelling, 0 for our labelling")
    # this acts on the number of classes (8 for users' labelling, 7 for our labelling)

    # comet
    parser.add_argument("--device", dest="device", default='0', help="choose GPU")
    parser.add_argument("--name_proj", dest="name_proj", default='MicroExpr', help="define comet ml project folder")
    parser.add_argument("--name_exp", dest="name_exp", default='None', help="define comet ml experiment")
    parser.add_argument("--comments", dest="comments", default=None, help="comments (str) about the experiment")

    parser.add_argument("--weights_path", dest="weights_path", default=None,
                        help="path to the folder where storing the model weights")

    parser.add_argument("--loss_weights", dest="loss_weights", default=0, help="1 to weight the loss, 0 otherwise")
    parser.add_argument("--data_aug", dest="data_aug", default=0,
                        help="1 to use augmented data, 0 otherwise. set to 1 if you pass train data with augmentation")


    args = parser.parse_args()

    # Hyper-parameters
    batch_size = int(args.batch_size)
    num_epochs = int(args.epochs)
    lr = float(args.lr)
    # wd = float(args.weight_decay)
    labelling = int(args.labelling)

    device = 'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    print("Device name: ", device, torch.cuda.get_device_name(int(args.device)))

    # Comet ml integration
    experiment = Experiment(project_name=args.name_proj)
    experiment.set_name(args.name_exp)

    # Definizione modello
    if labelling == 1:
        print("User labelling")
        num_classes = 8
        classes = ["Felicità", "Paura", "Sorpresa", "Disgusto", "Rabbia", "Tristezza", "Disprezzo", "Nessuna"]
    else:
        print("Our labelling")
        num_classes = 7
        classes = ["Felicità", "Paura", "Sorpresa", "Disgusto", "Rabbia", "Tristezza", "Disprezzo"]

    net = C3D(num_classes=num_classes)

    hyper_params = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "num_classes": num_classes,
        "loss_weights": int(args.loss_weights)
        # "weight_decay": wd,
    }

    experiment.log_parameters(hyper_params)
    experiment.set_model_graph(net)

    save_weights_path = os.path.join(args.weights_path, args.name_exp)
    if not os.path.exists(save_weights_path):
        os.makedirs(save_weights_path)
    print("save weights: ", save_weights_path)

    # Dataset, dataloaders

    f = open(args.train, 'rb')
    train_images, train_labels = pickle.load(f)
    f.close()

    f = open(args.valid, 'rb')
    valid_images, valid_labels = pickle.load(f)
    f.close()

    # questo solo perchè abbiamo i dati 'vecchi' che non sono tensori ... da togliere poi 
    if int(args.data_aug) == 1:
        # per i dati con data augmentation
        train_data = TensorDataset(train_images, train_labels)  # in load dataset salvo già i tensori
        val_data = TensorDataset(valid_images, valid_labels)
    else:
        train_data = TensorDataset(torch.from_numpy(train_images), torch.from_numpy(train_labels).type(torch.long))
        val_data = TensorDataset(torch.from_numpy(valid_images), torch.from_numpy(valid_labels).type(torch.long))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    validation_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size, drop_last=True)

    # Optimizer def
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # weight_decay=wd

    # Loss def
    if int(args.loss_weights) == 1:
        print("Loss weights")
        # pesi loss
        # class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
        #                                                                 classes=np.unique(train_labels), y=train_labels)
        # weight = torch.tensor(class_weights, dtype=torch.float).to(device)
        if labelling == 1:
            # 8 classes --> 8 weights
            weight = torch.tensor([2, 25, 1, 20, 15, 1, 10, 25]).float().to(device)
        else:
            # 7 classes --> 7 weights
            pass   # todo

        loss = torch.nn.CrossEntropyLoss(weight=weight)
    else:
        loss = torch.nn.CrossEntropyLoss()

    # Model to GPU
    net = net.to(device)

    # Parameters counter
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())  # [x for x in net.parameters()]
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of parameters: ", params)

    experiment.log_other('num_parameters', params)
    if args.comments:
        experiment.log_other('comments', args.comments)

    print("Start training loop")

    epoch_pc_acc = []
    for epoch in range(num_epochs):
        net.train()  # Sets the module in training mode

        # batch training
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        y_true = []
        y_pred = []

        for it, train_batch in enumerate(train_loader):
            train_images = train_batch[0].to(device)
            train_labels = train_batch[1].to(device)

            optimizer.zero_grad()

            # prediction
            out = net(train_images)  # (batch_size, num_classes)

            # compute loss
            train_loss = loss(out, train_labels)  # batch loss
            train_losses.append(train_loss.item())

            # compute accuracy
            train_acc = (out.argmax(dim=-1) == train_labels).float().mean()
            train_accuracies.append(train_acc.item())

            train_loss.backward()

            # update weights
            optimizer.step()

        # Validation step
        print()

        net.eval()  # Sets the module in evaluation mode (validation/test)
        with torch.no_grad():
            for val_it, val_batch in enumerate(validation_loader):
                val_images = val_batch[0].to(device)
                val_labels = val_batch[1].to(device)

                val_out = net(val_images)

                val_loss = loss(val_out, val_labels)
                val_losses.append(val_loss.item())

                val_acc = (val_out.argmax(dim=-1) == val_labels).float().mean()
                val_accuracies.append(val_acc.item())

                y_pred.extend(val_out.argmax(dim=-1).cpu().data.numpy())  # Save Prediction
                y_true.extend(val_labels.cpu().data.numpy())  # Save Truth

        # comet ml
        experiment.log_metric('train_epoch_loss', sum(train_losses) / len(train_losses), step=epoch + 1)
        experiment.log_metric('train_epoch_acc', sum(train_accuracies) / len(train_accuracies), step=epoch + 1)
        experiment.log_metric('valid_epoch_loss', sum(val_losses) / len(val_losses), step=epoch + 1)
        experiment.log_metric('valid_epoch_acc', sum(val_accuracies) / len(val_accuracies), step=epoch + 1)

        # confusion matrix
        # cf_train_mat = confusion_matrix(number_to_lab(train_labels),
        #                                 number_to_lab(out.argmax(dim=-1).cpu().data.numpy()), labels=classes)
        cf_valid_mat = confusion_matrix(number_to_lab(y_true),
                                        number_to_lab(y_pred), labels=classes)

        # plot_confusion_matrix(cf_train_mat, classes=classes,
        #                       normalize=True, step=epoch + 1, exp=experiment, title='train confusion matrix')

        plot_confusion_matrix(cf_valid_mat, classes=classes,
                              normalize=True, step=epoch + 1, exp=experiment, title='validation confusion matrix')

        # per class accuracy (of one epoch)
        per_class_val_acc = cf_valid_mat.diagonal() / cf_valid_mat.diagonal().sum()
        epoch_pc_acc.append(per_class_val_acc)

        print("Epoch [{}], Train loss: {:.4f}, Validation loss: {:.4f}".format(
            epoch + 1, sum(train_losses) / len(train_losses), sum(val_losses) / len(val_losses)))

        # Save weights
        if epoch % 10 == 0:
            torch.save(net.state_dict(), save_weights_path + '/weights_' + str(epoch + 1) + '.pth')

    torch.save(net.state_dict(), save_weights_path + '/final.pth')

    # log per-class-val-acc for all epochs
    epoch_pc_acc = np.array(epoch_pc_acc)
    # salvo in pickle per plottarla meglio a fine
    f = open(save_weights_path + '/epoch_pc_acc.pckl', 'wb')
    pickle.dump(epoch_pc_acc, f)
    f.close()
    # plt.clf()
    # fig, ax = plt.subplots(figsize=(num_epochs, num_classes))
    # sns.heatmap(epoch_pc_acc, annot=True, linewidths=.3)
    # # plt.imshow(epoch_pc_acc, cmap='hot', interpolation='nearest')
    # experiment.log_figure(figure_name='per-class-acc', figure=plt)

    experiment.end()
    print("End training loop")

    # Ex: python train.py --epochs 12 --batch_size 256 --name_exp vmr --weights_path ./model_weights
    # --comments 'first training'