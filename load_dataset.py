""" Load dataset from csv files """

import pandas as pd
import os
import numpy as np

IMAGE_HEIGHT = 200  # 112
IMAGE_WIDTH = 200  # 112
IMAGE_CHANNELS = 1
N_FRAMES = 51  # 50


def load_video(frames):
    # frames is an np.array with the path to the frames of a single video
    import matplotlib.pyplot as plt
    import cv2

    frames = frames.tolist()
    images = []  # accumula i frame
    for i in range(len(frames)):
        if os.path.exists(frames[i]):
            print("frame: ", frames[i])
            im = plt.imread(frames[i])  # np.array (h, w, c)
            print("im shape: ", im.shape)
            im = im[:, :, 0]  # tanto tutti i canali sono uguali (l'ultima slice è tutta di 1 ...opacità) (h, w)
            print("im shape: ", im.shape)
            # resize
            im = cv2.resize(im, dsize=(IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)  # dsize is w, h
            im = np.float32(im)
            images.append(im)

    images = np.array(images)  # fr, h, w
    images = images[np.newaxis, :]  # ch, fr, h, w
    print("images shape: ", images.shape)
    return images


def load_data(csv_path):
    data_df = pd.read_csv(csv_path, names=["user", "video", "frame", "label"], encoding='latin-1')
    # penso dia noia l'accento di felicità

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
            images = load_video(frames)  # np.array

            label = d1['label'].values[0]  # la prima label ... tanto è la stessa per tutti i frame del video
            label = lab_to_number(label)

            videos.append(images[:, 0:N_FRAMES, :, :])  # mi interessano solo i primi N_FRAMES
            labels.append(label)

            # if images.shape == (IMAGE_CHANNELS, N_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH):  # (1, 51, 200, 200)
            #     videos.append(images)
            #     labels.append(label)
            # else:
            #     print("video droppato (non posso fare il broadcast)")

    videos = np.array(videos)  # accumula i video come 'righe'  --> tenendo conto di questo ho fatto images a 4 dim
    labels = np.array(labels)
    print("final shape: ", videos.shape)
    print("final shape labels: ", labels.shape)

    return videos, labels


# todo: migliorare
def lab_to_number(lab):
    new_lab = 0  # Felicità
    if lab == 'Paura':
        new_lab = 1
    elif lab == 'Sorpresa':
        new_lab = 2
    elif lab == 'Disprezzo':
        new_lab = 3
    elif lab == 'Rabbia':
        new_lab = 4
    elif lab == 'Tristezza':
        new_lab = 5
    elif lab == 'Disprezzo':
        new_lab = 6
    elif lab == 'Nessuna':
        new_lab = 7

    return new_lab


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Load dataset train/validation")
    parser.add_argument("--csv_path", dest="save_path", default=None, help="path to the csv file to load "
                                                                           "dataset")
    parser.add_argument("--pckl_path", dest="save_path", default=None, help="path to the pickle output file")

    args = parser.parse_args()

    videos, labels = load_data(csv_path=args.csv_path)

    import pickle
    f = open(args.pckl_path, 'wb')
    pickle.dump([videos, labels], f)
    f.close()

    # csv_path = 'train2.csv'
    # videos, labels = load_data(csv_path)
    #
    # import pickle
    # f = open('video_labels.pckl', 'wb')
    # pickle.dump([videos, labels], f)
    # f.close()
