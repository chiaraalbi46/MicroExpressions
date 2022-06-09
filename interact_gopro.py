""" Manage video recording from event and go-pro camera """

import argparse
import os
import subprocess
import keyboard  # using module keyboard pip install keyboard
from goprocam import GoProCamera, constants


def rgb_recording(gopro, keyboard):
    while True:
        if keyboard.is_pressed('space'):
            print("space pressed --> stop the rgb recording")
            gopro.shutter(constants.stop)
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Record from event and rgb (go pro) cameras")

    parser.add_argument("--stream", dest="stream", default=None,
                        help="None for live strem from event camera, path to the a video otherwise")
    # 'C:/Users/chiar/PycharmProjects/Microexpressions/samples/monitoring_40_50hz.raw'

    parser.add_argument("--user", dest="user", default='user', help="user id")

    parser.add_argument("--event_dataset_path", dest="edp", default=None,
                        help="None if the event videos are stored in this folder, path to the dataset folder otherwise")
    parser.add_argument("--dataset_path", dest="dp", default=None,
                        help="path to the dataset folder otherwise")
    # parser.add_argument("--gopro_dataset_path", dest="gdp", default=None,
    #                     help="None if the gopro videos are stored in this folder,
    #                     path to the dataset folder otherwise")

    args = parser.parse_args()

    # Paths
    # if args.gdp is not None:
    #     gopro_records_path = './records/rgb/' + 'user_' + args.user
    # else:
    #     gopro_records_path = args.dp + '/gopro_videos/' + 'user_' + args.user  # .../gopro_videos/user1
    #
    # if not os.path.exists(gopro_records_path):
    #     print("Go pro record path")
    #     os.makedirs(gopro_records_path)

    if args.edp is not None:
        event_records_path = './records/event/' + 'user_' + args.user
    else:
        print("Event videos path")
        event_records_path = args.dp + '/event_videos/original/' + 'user_' + args.user
        # .../event_videos/original/user1

    if not os.path.exists(event_records_path):
        os.makedirs(event_records_path)

    # event camera
    player = "metavision_player"
    out_name = event_records_path + '/' + 'user' + args.user  # ./records/event/user_01/user1-timestamp.raw
    print("out name: ", out_name)

    # GoPro camera
    goproCamera = GoProCamera.GoPro(constants.gpcontrol)

    if args.stream is None:
        print("Live stream")
        command = [player, '--output-raw-basename', out_name]
    else:
        command = [player, '-i', args.stream, '--output-raw-basename', out_name]

    with subprocess.Popen(command, stdin=subprocess.PIPE) as p:
        while True:  # making a loop
            try:  # used try so that if user pressed other than the given key error will not be shown
                if keyboard.is_pressed('space'):
                    print("Start recording with Gopro cam")
                    goproCamera.shutter(constants.start)
                    # circa un secondo e 73 di ritardo per l'inizio della registrazione con la gopro
                    # goproCamera.shoot_video()
                    rgb_recording(goproCamera, keyboard)

                if keyboard.is_pressed('q'):  # if key 'q' is pressed
                    print('Stop the event camera and all the loop')
                    break  # finishing the loop

            except:
                break  # if user pressed a key other than the given key the loop will break ...

    # Download videos (this waste go pro battery)
    # medialist = goproCamera.listMedia(format=True, media_array=True)

    # count = 0
    # for media in medialist:
    #     if "MP4" in media[1]:
    #         # newpath = gopro_records_path + "/" + media[1]
    #         if count < 10:
    #             newpath = gopro_records_path + "/" + str(0) + str(count)
    #         else:
    #             newpath = gopro_records_path + "/" + str(count)
    #
    #         goproCamera.downloadMedia(media[0], media[1], newpath)
    #         count += 1

    # Delete videos from the go pro
    # goproCamera.delete('all')

    # Ex: python interact_gopro.py --user 00 --edp D:/Dataset_Microexpressions
