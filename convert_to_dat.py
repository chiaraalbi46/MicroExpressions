""" Convert the dataset video to .dat files, that can be read from the frame encoder """

import subprocess

# folder hierarchy ...

filename = './records/event/gopro_2022-05-30_09-23-15.raw'  # raw video
# out_name = './pino'
command = ['metavision_raw_to_dat', '-i', filename]
#  '--output-raw-basename', out_name non funziona
subprocess.Popen(command, stdin=subprocess.PIPE)

# il file di default viene salvato nella stessa cartella del file da convertire, con lo stesso nome e alla fine _cd.dat
# dopo andranno rinominati e raccolti nella cartella 'giusta'
