""" Try to record the user while he is watching a video displayed on the screen """

import time
import cv2
import os


def write_frame(vid, writer):
    ret, frame = vid.read()

    writer.write(frame)

    winname = "Recording"
    cv2.namedWindow(winname)  # Create a named window
    # cv2.moveWindow(winname, 1400, 0)  # Move it to 0, 0
    cv2.moveWindow(winname, 1000, 0)
    cv2.imshow(winname, frame)


def display_video():
    # se interrompo il video si deve interrompere anche la registrazione !
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        # if ret:
        #     frame = cv2.resize(frame, (1280, 720))
        # else:
        #     break
        winname = "Video"
        cv2.namedWindow(winname)  # Create a named window
        cv2.moveWindow(winname, 0, 0)  # Move it to 0, 0
        cv2.imshow(winname, frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            print("stop the video streaming")
            write_frame(vid, writer)  # registro l'ultimo frame...  # todo: capire se qui va bene ...
            end = time.time()
            print("time to time display the video with manual stop: ", end-start)
            return 1

        # continue updating the recording
        write_frame(vid, writer)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break  # esce dal loop su cap, ma non da quello esterno

    end = time.time()
    print("time to time display the video: ", end - start)
    return 0

# Paths
records_path = './records'  # here we store the user reaction from the coau cam (rgb)
if not os.path.exists(records_path):
    os.makedirs(records_path)

videos_path = './videos'  # here we have the video to show the users in order to get the microexpressions
if not os.path.exists(videos_path):
    os.makedirs(videos_path)

# RGB video capture
vid = cv2.VideoCapture(0)  # 1 per la coau, 0 per la camera interna del computer

width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(records_path + '/capture.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))

# Video to show to get microexpression
cap = cv2.VideoCapture(videos_path + '/video1.mp4')

start_displaying = 5  # tempo dopo cui attivare il video da mostrare

import subprocess
first_start = time.time()

# event camera
# player = "metavision_player"
# subprocess.Popen([player], stdin=subprocess.PIPE)
# video_path = 'C:/Users/chiar/PycharmProjects/Microexpressions/samples/monitoring_40_50hz.raw'
# save_path = 'C:/Users/chiar/PycharmProjects/Microexpressions/bo'
# subprocess.Popen([player, '-i', video_path, '--output-raw-basename', save_path], stdin=subprocess.PIPE)
while True:
    # start = time.time()
    write_frame(vid, writer)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # end = time.time()
    # print("time to write a frame: ", end-start)

    current_time = time.time()
    print("current time: ", current_time - first_start)
    if current_time - first_start >= start_displaying:
        print("Start the video streaming")
        out = display_video()
        if out == 0:
            print("Video and record completed !")
            break
        else:
            print("Video streaming (and recording) interrupted !")
            break

# After the loop release the cap object
cap.release()
vid.release()
writer.release()

cv2.destroyAllWindows()