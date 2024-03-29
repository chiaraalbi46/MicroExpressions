""" Load dataset from csv files """

import pandas as pd
import os
import numpy as np
import torch

IMAGE_HEIGHT = 200  # 112
IMAGE_WIDTH = 200  # 112
IMAGE_CHANNELS = 1
N_FRAMES = 51  # 50


def load_video(frames, data_aug):
    # frames is an np.array with the path to the frames of a single video
    import matplotlib.pyplot as plt
    import cv2
    from PIL import Image
    import torchvision.transforms as transforms
    import torch

    if data_aug == 1:
        print("Data augmentation")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(0, 5))])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=transforms.InterpolationMode.NEAREST)])

    frames = frames.tolist()
    images = []  # accumula i frame
    for i in range(len(frames)):
        if os.path.exists(frames[i]):
            print("frame: ", frames[i])
            # im = plt.imread(frames[i])  # np.array (h, w, c)
            # print("im shape: ", im.shape)
            # im = im[:, :, 0]  # tanto tutti i canali sono uguali (l'ultima slice è tutta di 1 ...opacità) (h, w)
            # # print("im shape: ", im.shape)

            ##
            im = Image.open(frames[i])
            im.load()
            im = np.array(im)[:, :, 0]
            im = Image.fromarray(im, mode='L')
            im = transform(im)
            ##

            # resize
            # im = cv2.resize(im, dsize=(IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)  # dsize is w, h
            # im = np.float32(im)
            images.append(im)

    # images = np.array(images)  # fr, h, w
    # images = images[np.newaxis, :]  # ch, fr, h, w

    images = torch.stack(images, 0)  # fr, ch, h, w
    images = images.permute(1, 0, 2, 3)  # ch, fr, h, w
    print("images shape: ", images.shape)
    return images


def load_data(csv_path, data_aug):
    data_df = pd.read_csv(csv_path, names=["user", "video", "frame", "label"])

    users = data_df['user'].values
    u = np.unique(users)

    videos = []
    labels = []
    for us in u:
        print('User: ', us)
        d = data_df.loc[data_df['user'] == us]  # blocco dell'utente corrente
        v = d['video'].values
        v_u = np.unique(v)  # np.array dei video dell'utente corrente
        for vid in v_u:
            d1 = d.loc[d['video'] == vid]  # blocco del video corrente
            frames = d1['frame'].values  # N_FRAMES per video
            images = load_video(frames, data_aug=data_aug)  # np.array

            label = d1['label'].values[0]  # la prima label ... tanto è la stessa per tutti i frame del video
            label = lab_to_number(label)

            videos.append(images[:, 0:N_FRAMES, :, :])  # mi interessano solo i primi N_FRAMES
            labels.append(label)

    print("videos shape: ", len(videos))
    # videos = np.array(videos)  # accumula i video come 'righe'  --> tenendo conto di questo ho fatto images a 4 dim
    labels = np.array(labels)
    labels = torch.from_numpy(labels).type(torch.long)

    videos = torch.stack(videos, 0)

    print("final shape: ", videos.shape)
    print("final shape labels: ", labels.shape)

    return videos, labels


def lab_to_number(lab):
    if lab == 'Felicità':
        new_lab = 0
    elif lab == 'Paura':
        new_lab = 1
    elif lab == 'Sorpresa':
        new_lab = 2
    elif lab == 'Disgusto':
        new_lab = 3
    elif lab == 'Rabbia':
        new_lab = 4
    elif lab == 'Tristezza':
        new_lab = 5
    elif lab == 'Disprezzo':
        new_lab = 6
    else:
        new_lab = 7  # 'Nessuna'

    return new_lab


def number_to_lab(numbers):
    labels = []
    for number in numbers:
        if number == 0:
            new_lab = 'Felicità'
        elif number == 1:
            new_lab = 'Paura'
        elif number == 2:
            new_lab = 'Sorpresa'
        elif number == 3:
            new_lab = 'Disgusto'
        elif number == 4:
            new_lab = 'Rabbia'
        elif number == 5:
            new_lab = 'Tristezza'
        elif number == 6:
            new_lab = 'Disprezzo'
        else:
            new_lab = 'Nessuna'  # number = 7

        labels.append(new_lab)

    return np.asarray(labels)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Load dataset train/validation")
    parser.add_argument("--csv_path", dest="csv_path", default=None, help="path to the csv file to load dataset")
    parser.add_argument("--pckl_path", dest="pckl_path", default=None, help="path to the pickle output file")
    parser.add_argument("--data_aug", dest="data_aug", default=0, help="1 to apply data augmentation, 0 otherwise")

    args = parser.parse_args()

    videos, labels = load_data(csv_path=args.csv_path, data_aug=int(args.data_aug))

    import pickle

    f = open(args.pckl_path, 'wb')
    pickle.dump([videos, labels], f)
    f.close()

    # import torch
    # labels_num = np.float32(np.array((0, 1, 0, 2, 1)))
    # t = torch.from_numpy(labels_num).type(torch.long)
    # res = number_to_lab(t.numpy())

    # csv_path = 'train2.csv'
    # videos, labels = load_data(csv_path)
    #
    # import pickle
    # f = open('video_labels.pckl', 'wb')
    # pickle.dump([videos, labels], f)
    # f.close()

    ####
    # csv_path = 'C:/Users/chiar/Desktop/train_short.csv'
    # data_df = pd.read_csv(csv_path, names=["user", "video", "frame", "label"])
    # users = data_df['user'].values
    # u = np.unique(users)
    #
    # d = data_df.loc[data_df['user'] == u[0]]  # blocco dell'utente corrente
    # v = d['video'].values
    # v_u = np.unique(v)  #
    # d1 = d.loc[d['video'] == v_u[0]]  # blocco del video corrente
    #
    # label = d1['label'].values[0]
    # label = label.replace(label[-2:], 'à')
    #
    # frames = d1['frame'].values  # N_FRAMES per video
    # images = load_video(frames)  #
