import pickle
import librosa
import matplotlib.pyplot as plt
import IPython.display
import librosa.display
import itertools
import numpy as np
import csv
import os
from pandas import read_csv


class Feature:

    hop_length = 512

    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(os.path.normpath(filepath)).split(".")[0]
        self.y, self.sr = librosa.load(path=filepath, sr=22050, offset=15)