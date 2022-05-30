""" The same code of rgb_event_recording but recording using go pro directly """

import time
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

    parser.add_argument("--user", dest="user", default=None, help="user id")  # ...
    parser.add_argument("--event_dataset_path", dest="edp", default=None, help="None if the event videos are stored in "
                                                                               "this folder, path to the dataset folder"
                                                                               "otherwise")

    args = parser.parse_args()

    # Paths
    # rgb_records_path = './records/rgb'  # here we store the user reaction from the rgb cam
    # if not os.path.exists(rgb_records_path):
    #     os.makedirs(rgb_records_path)

    if args.edp is None:
        event_records_path = './records/event'  # here we have the user reaction from the event cam
    else:
        event_records_path = args.edp + '/event'

    if not os.path.exists(event_records_path):
        os.makedirs(event_records_path)

    # event camera
    player = "metavision_player"
    if args.user is not None:
        out_name = event_records_path + '/' + args.user
    else:
        out_name = event_records_path + '/user'

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

