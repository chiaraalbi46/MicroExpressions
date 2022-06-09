""" Working with conversion of event camera stream in frames """

import numpy as np
import sys
sys.path.insert(0, '../prophesee-automotive-dataset-toolbox/')
from src.io.psee_loader import PSEELoader


# open a file
video = PSEELoader("C:/Users/chiar/OneDrive/Documents/Prophesee/out_2022-05-20_13-03-10_cd.dat")  # dat file
print(video)  # show some metadata
video.event_count()  # number of events in the file
video.total_time()  # duration of the file in mus

# look at https://github.com/fedebecat/tbr-event-object-detection/blob/master/src/event_converter.py


def show_image(frame: np.array, max_value: int = 1):
    """
    @brief: show video of encoded frames and their bboxes
            during processing
    @param: frame - A np array containing pixel informations
    @param: bboxes - np array with the bboxes associated to the frame.
                     As loaded from the GEN1 .npy array
    """

    plt.figure(1)
    plt.clf()
    plt.axis("off")
    plt.imshow(frame, animated=True, cmap='gray', vmin=0, vmax=max_value)


# Load
from tbe import TemporalBinaryEncoding
import matplotlib.pyplot as plt
from encoders import *

gen1_video = video  # PSEELoader(video_path + "_td.dat")

width = gen1_video.get_size()[1]
height = gen1_video.get_size()[0]
## mettere 'giusti'
tbr_bits = 5
delta_t = 50000
##
encoder = TemporalBinaryEncoding(tbr_bits, width, height)
encoded_array = encode_video_tbe(tbr_bits, width, height, gen1_video, encoder, delta_t)

# Iterate through video frames
img_count = 0
bbox_count = 0
print("Saving encoded frames and bounding boxes...")
for f in encoded_array:

    # filename = video_name + str("_" + str(f["startTs"]))
    # Save images that have at least a bbox

    # Save image
    # plt.imsave(dir_paths["images"] + "/" + filename + ".jpg", f['frame'], vmin=0, vmax=1, cmap='gray')

    img_count += 1

    # if show_video:
    show_image(f['frame'])
    plt.show()
    plt.pause(0.05)

    # if export_all_frames_requested:
    # save_bb_image(f['frame'], np.array([]), export_frames_path + filename + "_" + requested_encoder + ".jpg", False,
    #               max_pixel_value)

# print("Saved {:d} encoded frames in path: {:s}".format(img_count, dir_paths["images"]))
