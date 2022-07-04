""" Create csv file for dataloaders """

import csv
import os
import pandas as pd


# i csv dovranno avere utente, video, path al frame, label (emozione)
# se seguiamo l'etichettatura dei moduli dovremo prendere i moduli esportarli in csv ed estrarre per ogni utente,
# per ogni video le emozioni da lui scelte
# tutti i frame di uno stesso video verrano etichettati allo stesso modo

# per scrivere i csv mi basavo su uno split del dataset in train, valid e test, per ognuno c'era un file associato
#  conteneva i video di train etc ... nel nostro caso lo split si farebbe per utenti
# simuliamo questa cosa con una lista che contiene gli utenti da usare per questa prova ...


def create_csv(base_folder, users_list, emotion_csv, save_path):
    users = sorted(os.listdir(base_folder))  # sorted is important because linux doesn't follow alphabetical order

    # se ho una matrice utenti x video
    df = pd.read_csv(emotion_csv)
    users_video_labels = df.values[:, np.arange(4, len(df.columns), 2)]  # 29, 21
    ids = df.values[:, 3] - 1  # parto da 0
    surnames = df.values[:, 2]

    with open(save_path, 'w') as csvfile:  # ciclo sugli utenti
        filewriter = csv.writer(csvfile)
        for i in range(len(users)):
            user_path = base_folder + users[i] + '/'  # .../user_00/
            videos = sorted(os.listdir(user_path))  # lista cartelle video
            # users[i][-2:] --> le ultime due lettere di user_00
            if users[i][-2:] in users_list:  # train/validation
                if users[i][-2:-1] == '0':
                    uid = int(users[i][-1])  # l'ultima cifra è l'utente
                else:
                    uid = int(users[i][-2:])
                print('users[i], uid: ', users[i], uid)
                ind = np.where(ids == uid)  # tupla ...
                print("indice riga dell'id utente scelto: ", ind[0][0])
                # per ora tutti i video di utente stanno insieme (o train o validation)
                for j in range(len(videos)):  # ciclo sulle cartelle dei video dell'utente
                    video_dir = videos[j]
                    print("video dir: ", video_dir)
                    video_dir_path = user_path + video_dir + '/'  # path alla cartella video j-esimo

                    # lab = users_video_labels[uid][j]  # utente, video
                    # sto assumendo implicitamente che l'ordine degli id degli utenti sia coerente ... ma non è così ..
                    # ad esempio ad uid 18 corrisponde l'id 19 dell'utente ...

                    print('surname: ', surnames[ind[0][0]])
                    lab = users_video_labels[ind[0][0]][j]  # utente, video
                    print('label: ', lab)
                    print('')

                    frames = sorted(os.listdir(video_dir_path))
                    # etichetto tutti i frame con la label del video cui appartengono
                    for k in range(len(frames)):  # ciclo sui frame del video j
                        frame_path = video_dir_path + frames[k]

                        # line = [user_path, video_dir_path, frame_path, lab]
                        line = ['user_' + users[i][-2:], video_dir, frame_path, lab]
                        filewriter.writerow(line)


if __name__ == '__main__':
    import numpy as np
    import argparse

    parser = argparse.ArgumentParser(description="Create csv file for training/validation/test")

    parser.add_argument("--dataset_folder", dest="dataset_folder", default=None, help="dataset of frames")  # end slash
    parser.add_argument("--emotion_csv_path", dest="emotion_csv_path", default='Progetto VMR - Microespressioni.csv',
                        help="csv to label videos")
    parser.add_argument("--save_path", dest="save_path", default=None, help="path to the output csv file")

    parser.add_argument("--train", dest="train", default=1, help="1 train, 0 validation")

    args = parser.parse_args()

    train_users_list = ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '14', '15', '16',
                        '21', '22', '23', '24', '25', '26']  # 20 utenti
    validation_users_list = ['07', '19']  # 2 utenti
    # todo: sarà random poi ... o scritta in un altro file che si può passare per estrarla

    var = int(args.train)
    if var == 1:
        print("Train csv")
        create_csv(base_folder=args.dataset_folder, users_list=train_users_list, emotion_csv=args.emotion_csv_path,
                   save_path=args.save_path)
    else:
        print("Validation csv")
        create_csv(base_folder=args.dataset_folder, users_list=validation_users_list, emotion_csv=args.emotion_csv_path,
                   save_path=args.save_path)

    # Ex: python create_csv_file.py --dataset_folder /home/calbisani/event_frame_NEW/ --save_path train.csv

    # base_folder = 'D:/Dataset_Microexpressions/frame_dataset/'  # cartella con i frame
    # save_path = 'train2.csv'
    # emotion_csv_path = 'Progetto VMR - Microespressioni.csv'  # csv per etichettatura video
    #
    # create_csv(base_folder, train_users_list, emotion_csv_path, save_path)



