from Feature import Feature
import itertools
import numpy as np
import csv
import time
import matplotlib.pyplot as plt
import os, errno
from pandas import read_csv

# get audio files for feature extraction
def get_audio_directory():
    cd = os.path.dirname(os.path.realpath(__file__))
    resource_dir = cd + '/resource/music/'
    music_dir = [resource_dir + f for f in os.listdir(resource_dir)]
    return music_dir

# get the annotation
def get_labels():
    arousal = read_csv("resource/music_label/arousal.csv", header=0)
    valence = read_csv("resource/music_label/valence.csv", header=0)
    return arousal, valence


def get_feature_name(name, total_round=1):
    i = 0
    new_name = []
    if total_round == 1:
        new_name.append(name)
    else:
        while i < total_round:
            new_name.append(name + "_" + str(i))
            i += 1
    return new_name


# create dir for storing features
try:
    os.mkdir('resource')
    os.mkdir('resource/features')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# get directory for all the music
musicDir = get_audio_directory()
# get labels for the music
arousal, valence = get_labels()
# get all features' header
all_feature_name = []
all_feature_name.append(get_feature_name("mfcc", 13))
all_feature_name.append(get_feature_name("mfcc_delta", 13))
all_feature_name.append(get_feature_name("mfcc_delta2", 13))
all_feature_name.append(get_feature_name("spectral_flux"))
all_feature_name.append(get_feature_name("zero_crossing_rate"))
all_feature_name.append(get_feature_name("rolloff95"))
all_feature_name.append(get_feature_name("rolloff85"))
all_feature_name.append(get_feature_name("spectral contrast", 7))
all_feature_name.append(get_feature_name("spectral_centroid"))
all_feature_name.append(get_feature_name("chroma", 12))
all_feature_name.append(get_feature_name("tonnetz", 6))
all_feature_name.append(get_feature_name("rmse"))
all_feature_name.append(get_feature_name("energy_novelty"))
all_feature_name.append(get_feature_name("loudness"))
all_feature_name.append(get_feature_name("dtempo"))
all_feature_name = list(itertools.chain.from_iterable(all_feature_name))
all_feature_name_mean = [(x + '_mean') for x in all_feature_name]
all_feature_name_std = [(x + '_std') for x in all_feature_name]
features_header = []
features_header.append(all_feature_name_mean)
features_header.append(all_feature_name_std)
features_header.append(get_feature_name("arousal"))
features_header.append(get_feature_name("valence"))
features_header = list(itertools.chain.from_iterable(features_header))
i = 0
for music in musicDir:

    start = time.time()
    # extract all features
    features = Feature(music, 15)
    print("Extracting music feature: ", features.filename, "...")
    print("extracting timbre...")
    features.extract_timbre_features()
    print("extracting melody...")
    features.extract_melody_features()
    plt.show()
    print("extracting energy...")
    features.extract_energy_features()
    print("extracting rhythm...")
    features.extract_rhythm_features()
    all_features = features.get_all_features()

    # get current feature labels
    row_arousal_label = arousal.loc[arousal['song_id'] == int(features.filename)].index[0]
    row_valence_label = valence.loc[valence['song_id'] == int(features.filename)].index[0]
    curr_arousal_label = arousal.values[row_arousal_label, 1:]
    curr_valence_label = valence.values[row_valence_label, 1:]
    # remove nan from list
    curr_arousal_label = curr_arousal_label[~np.isnan(curr_arousal_label)]
    curr_valence_label = curr_valence_label[~np.isnan(curr_valence_label)]
    # removing excess label/timestamps
    min_music_length = min(len(curr_arousal_label), len(curr_valence_label))
    all_features = all_features[:, :min_music_length]
    curr_arousal_label = curr_arousal_label[:min_music_length]
    curr_valence_label = curr_valence_label[:min_music_length]
    # append label to the data
    all_features = np.vstack([all_features, curr_arousal_label, curr_valence_label])

    # save feature to csv
    with open("resource/features/" + features.filename + ".csv", "w+", newline='') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerow(features_header)
        # invert feature to be (timestamp x feature) format
        csvWriter.writerows(np.transpose(all_features))
    end = time.time()
    i += 1
    print(features.filename, " done, total left: ", i, "/", len(musicDir))
    print("Runtime: ", end-start, " seconds")

    print('done')