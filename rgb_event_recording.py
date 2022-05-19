""" Automatic recording of th rgb and the event camera """

import time
import cv2
import os
import subprocess
import keyboard  # using module keyboard pip install keyboard

# todo: main e argparse
# todo: calcolare un po' i ritardi
# todo: settare il path per le registrazioni


def write_frame(vid, writer):
    ret, frame = vid.read()

    writer.write(frame)

    winname = "Recording"
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, 1400, 0)  # Move it to 0, 0
    cv2.imshow(winname, frame)


def rgb_recording(vid, writer, keyboard):
    while True:
        write_frame(vid, writer)
        # if cv2.waitKey(1) & 0xFF == 32:  # ord('q')
        if cv2.waitKey(1) & keyboard.is_pressed('space'):
            print("space pressed --> stop the rgb recording")
            break

    vid.release()
    writer.release()

    cv2.destroyAllWindows()


# Paths
rgb_records_path = './records/rgb'  # here we store the user reaction from the rgb cam
if not os.path.exists(rgb_records_path):
    os.makedirs(rgb_records_path)

event_records_path = './records/event'  # here we have the user reaction from the event cam
if not os.path.exists(event_records_path):
    os.makedirs(event_records_path)

# event camera
player = "metavision_player"
video_path = 'C:/Users/chiar/PycharmProjects/Microexpressions/samples/monitoring_40_50hz.raw'
save_path = 'C:/Users/chiar/PycharmProjects/Microexpressions/ciao2'

# subprocess.Popen([player], stdin=subprocess.PIPE)  # to open the event camera live stream

with subprocess.Popen([player, '-i', video_path, '--output-raw-basename', save_path], stdin=subprocess.PIPE) as p:
    while True:  # making a loop
        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('space'):
                print("Start recording with RGB cam")
                vid = cv2.VideoCapture(0)
                width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                writer = cv2.VideoWriter(rgb_records_path + '/basicvideo2.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20,
                                         (width, height))
                rgb_recording(vid, writer, keyboard)
            if keyboard.is_pressed('q'):  # if key 'q' is pressed
                print('Stop the event camera and all the loop')
                break  # finishing the loop

        except:
            break  # if user pressed a key other than the given key the loop will break ...
