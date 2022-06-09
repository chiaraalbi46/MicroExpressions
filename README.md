![GitHub contributors](https://img.shields.io/github/contributors/chiaraalbi46/EvasiveMovements?color=blue) ![GitHub repo size](https://img.shields.io/github/repo-size/chiaraalbi46/EvasiveMovements) [![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/) ![EventCamera](https://img.shields.io/badge/Metavision%20SDK-2.3.2-blue) 

# MicroExpressions
Catch people micro-expressions using an RGB camera and an Event camera. Then try to classify them: which is the best ? 

## Frame coding 
Event camera capture a stream of events. This stream needs to be 'converted' in frames, in order to deal with neural network processing.
Different codings are possible, here we work with *Temporal Binary Encoding*. Please look at [link](https://github.com/fedebecat/tbr-event-object-detection) to get more information about the coding. 
Here we install only the necessary files. The **prophesee-automotive-dataset-toolbox** has been downloaded from [link](https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox) and added to the sys path so
it can be used as a project module. Folder **frame_coding** contains encoders.py and tbe.py from [link](https://github.com/fedebecat/tbr-event-object-detection/tree/master/src) and the files related 
to the conversion of the video streams of the dataset in frames. 

## Proposed folder hierarchy

```
dataset                                    
│
├── event_videos
│    ├── original
│        ├── user_01
│            ├── user_01-timestamp.raw
│            ├── user_01-timestamp.raw
│            ├── user_01-timestamp.raw
│            ├── user_01-timestamp.raw
│            └── ...
│    ├── cut
│        ├── user_01
│            ├── user_01-timestamp.raw
│            ├── user_01-timestamp.raw
│            ├── user_01-timestamp.raw
│            ├── user_01-timestamp.raw
│            └── ...
│ 
├── gopro_videos
│   ├── user_01
│   ├── user_02
│   └── ...
│
├── dat_dataset
│   └── ...
├── frame_dataset
│   └── ...
├── csv_dataset  # uno per utente
│   └── 
└──  
```