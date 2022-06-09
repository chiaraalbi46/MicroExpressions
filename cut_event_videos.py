""" Interface to cut videos using metavision_raw_cutter utility """

# todo: ciclo seguendo la gerarchia di cartelle

import subprocess

path_video = './samples/monitoring_40_50hz.raw'
start = 2.0
end = 4.0
cut_path_video = './samples/ciao.raw'
subprocess.Popen(['metavision_raw_cutter', '-i', path_video, '-s', str(start), '-e', str(end), '-o', cut_path_video])