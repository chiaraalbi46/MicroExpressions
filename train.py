""" Training loop using C3D model """

from comet_ml import Experiment
import os.path
import torch
# from load_dataset import load_data
import pickle
from torch.utils.data import TensorDataset, DataLoader
import argparse
import numpy as np
from C3D_model import C3D


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train C3D")

    parser.add_argument("--epochs", dest="epochs", default=1, help="number of epochs")
    parser.add_argument("--batch_size", dest="batch_size", default=256, help="Batch size")
    parser.add_argument("--lr", dest="lr", default=0.001, help="learning rate train")
    # parser.add_argument("--weight_decay", dest="weight_decay", default=0., help="weight decay")
    # parser.add_argument("--val_perc", dest="val_perc", default=0, help="% validation set")

    # iperparametri C3D

    # train/valid pickle
    parser.add_argument("--train", dest="train", default=None, help="path to train pickle file")
    parser.add_argument("--valid", dest="valid", default=None, help="path to validation pickle file")

    # comet
    parser.add_argument("--device", dest="device", default='0', help="choose GPU")
    parser.add_argument("--name_proj", dest="name_proj", default='MicroExpr', help="define comet ml project folder")
    parser.add_argument("--name_exp", dest="name_exp", default='None', help="define comet ml experiment")
    parser.add_argument("--comments", dest="comments", default=None, help="comments (str) about the experiment")

    parser.add_argument("--weights_path", dest="weights_path", default=None,
                        help="path to the folder where storing the model weights")

    args = parser.parse_args()

    # Hyper-parameters
    batch_size = int(args.batch_size)
    num_epochs = int(args.epochs)
    lr = float(args.lr)
    # wd = float(args.weight_decay)

    device = 'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    print("Device name: ", device, torch.cuda.get_device_name(int(args.device)))

    hyper_params = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        # "weight_decay": wd,  # todo aggiungi iperparametri c3d
    }

    # Comet ml integration
    experiment = Experiment(project_name=args.name_proj)
    experiment.set_name(args.name_exp)

    # Definizione modello
    net = C3D()

    experiment.log_parameters(hyper_params)
    experiment.set_model_graph(net)

    save_weights_path = os.path.join(args.weights_path, args.name_exp)
    if not os.path.exists(save_weights_path):
        os.makedirs(save_weights_path)
    print("save weights: ", save_weights_path)

    # # save hyperparams dictionary in save_weights_path
    # with open(save_weights_path + '/hyperparams.json', "w") as outfile:
    #     json.dump(hyper_params, outfile, indent=4)

    # Dataset, dataloaders

    # train_images, train_labels = load_data(csv_path=args.train)
    # valid_images, valid_labels = load_data(csv_path=args.valid)

    f = open(args.train, 'rb')
    train_images, train_labels = pickle.load(f)
    f.close()

    f = open(args.valid, 'rb')
    valid_images, valid_labels = pickle.load(f)
    f.close()

    train_data = TensorDataset(torch.from_numpy(train_images), torch.from_numpy(train_labels).type(torch.long))
    val_data = TensorDataset(torch.from_numpy(valid_images), torch.from_numpy(valid_labels).type(torch.long))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    validation_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size, drop_last=True)

    # Optimizer def
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # weight_decay=wd

    # Loss def
    loss = torch.nn.CrossEntropyLoss()  # dovrebbe andar bene dato che è un problema di classificazione multilabel

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

    for epoch in range(num_epochs):
        net.train()  # Sets the module in training mode

        # batch training
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
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

        # comet ml
        experiment.log_metric('train_epoch_loss', sum(train_losses) / len(train_losses), step=epoch + 1)
        experiment.log_metric('train_epoch_acc', sum(train_accuracies) / len(train_accuracies), step=epoch + 1)
        experiment.log_metric('valid_epoch_loss', sum(val_losses) / len(val_losses), step=epoch + 1)
        experiment.log_metric('valid_epoch_acc', sum(val_accuracies) / len(val_accuracies), step=epoch + 1)

        # print("End valid test")
        print("Epoch [{}], Train loss: {:.4f}, Validation loss: {:.4f}".format(
            epoch + 1, sum(train_losses) / len(train_losses), sum(val_losses) / len(val_losses)))

        # Save weights
        if epoch % 10 == 0:
            torch.save(net.state_dict(), save_weights_path + '/weights_' + str(epoch + 1) + '.pth')

    torch.save(net.state_dict(), save_weights_path + '/final.pth')

    experiment.end()
    print("End training loop")

    # Ex: python train.py --epochs 12 --batch_size 256 --name_exp vmr --weights_path ./model_weights
    # --comments 'first training'