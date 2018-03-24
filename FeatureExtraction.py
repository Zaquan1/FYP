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

sr = 22050
hop_length = 512


def get_directory():
    with open('music.pkl', 'rb') as f:
        music_dir = pickle.load(f)
    return music_dir


def get_labels():
    arousal = read_csv("resource/music_label/arousal.csv", header=0)
    valence = read_csv("resource/music_label/valence.csv", header=0)
    return arousal, valence


def duration_to_frame(y, idealLength):
    duration = librosa.get_duration(y=y, sr=sr, hop_length=hop_length)
    sec_inc = 0.5
    time_stamp = sec_inc
    time = []
    while time_stamp <= duration:
        time.append(time_stamp)
        time_stamp += sec_inc
    while len(time) > idealLength-1:
        time = time[:-1]
    return librosa.time_to_frames(time, hop_length=hop_length, sr=sr)


def sync_frames(features, frames, aggregate=np.mean):
    sync_feature = librosa.util.sync(features, frames, aggregate=aggregate)
    return sync_feature


# mfcc represent timbre
def get_mfcc_feature(y):
    y_harmonic = librosa.effects.harmonic(y, margin=4)
    # get mfcc
    mfcc = librosa.feature.mfcc(y=y_harmonic, sr=sr, hop_length=hop_length, n_mfcc=13)
    # get dmfcc
    mfcc_delta = librosa.feature.delta(mfcc)
    # get ddmfcc
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    # get spectral flux
    spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)
    spectral_flux = spectral_flux/spectral_flux.max()
    # get zero crossing rate
    zc = librosa.feature.zero_crossing_rate(y=y)
    # get rolloff of 95% and 85%
    rolloff95 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)
    rolloff85 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    # get spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y, sr=sr)
    # get spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y, sr=sr)
    '''
    # plot for rolloff
    plt.figure()
    plt.subplot(4,1,1)
    plt.semilogy(rolloff85.T)
    plt.xticks([])
    plt.xlim([0, rolloff85.shape[-1]])
    plt.subplot(4, 1, 2)
    plt.semilogy(rolloff95.T)
    plt.subplot(4, 1, 3)
    plt.semilogy((sync_frames(rolloff95, duration_to_frame(y))).T)
    plt.subplot(4, 1, 4)
    plt.semilogy((sync_frames(rolloff95, duration_to_frame(y), np.std)).T)
    
    #plot for mfcc
    plt.figure(1, figsize=(12, 6))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(mfcc)
    plt.ylabel('MFCC')
    plt.colorbar()
    plt.subplot(2, 1, 2)
    librosa.display.specshow(mfcc_delta)
    plt.ylabel('MFCC-$\Delta$')
    plt.colorbar()
    
    #plot for flux
    plt.figure(2)
    plt.subplot(4,1,1)
    plt.plot(2 + spectral_flux, alpha=0.8, label='Mean (mel)')
    plt.subplot(4, 1, 2)
    plt.plot(2 + spectral_flux2, alpha=0.8, label='Mean (mel)')
    plt.subplot(4,1,3)
    plt.plot(2 + spectral_flux/spectral_flux.max(), alpha=0.8, label='Mean (mel)')
    plt.subplot(4, 1, 4)
    plt.plot(2 + spectral_flux2/spectral_flux2.max(), alpha=0.8, label='Mean (mel)')
    '''
    return np.vstack([mfcc, mfcc_delta, mfcc_delta2, spectral_flux,
                      zc, rolloff95, rolloff85, spectral_contrast, spectral_centroid])


# chroma represent pitch/melody
def get_chroma_feature(y):
    y_harmonic = librosa.effects.harmonic(y, margin=4)
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=hop_length, bins_per_octave=12*3)
    # get tonal centroid
    tonnets = librosa.feature.tonnetz(y_harmonic, sr=sr)
    '''''
    plt.figure()
    plt.subplot(2,1,1)
    librosa.display.specshow(tonnets)
    plt.colorbar()
    plt.title('tonnetzz')
    plt.subplot(2, 1, 2)
    librosa.display.specshow(chromagram)
    plt.colorbar()
    plt.title('Chroma')
    plt.tight_layout()
    '''
    return np.vstack([chromagram, tonnets])


# rmse represent rmsEnergy
def get_rmse_feature(y):
    y_stft = librosa.stft(y=y, hop_length=hop_length)
    S, phase = librosa.magphase(y_stft)
    rms = librosa.feature.rmse(S=S).flatten()
    rms_diff = np.zeros_like(rms)
    rms_diff[1:] = np.diff(rms)
    energy_novelty = np.max([np.zeros_like(rms_diff), rms_diff], axis=0)
    # get loudness
    power = np.abs(S)**2
    p_mean = np.sum(power, axis=0, keepdims=True)
    p_ref = np.max(power)
    loudness = librosa.power_to_db(p_mean, ref=p_ref)
    '''
    # energy plot
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(sync_frames(rms.T, duration_to_frame(y)))
    plt.subplot(3,1,2)
    plt.plot(sync_frames(rms_diff.T, duration_to_frame(y)))
    plt.subplot(3, 1, 3)
    plt.plot(sync_frames(energy_novelty.T, duration_to_frame(y)))
    
    '''
    return np.vstack([rms, energy_novelty, loudness])


def get_rhythm_feature(y):
    onset = librosa.onset.onset_strength(y, sr=sr)
    dynamic_tempo = librosa.beat.tempo(onset_envelope=onset, sr=sr, aggregate=None)
    '''
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(dynamic_tempo, linewidth=1.5, label='tempo estimate')
    plt.legend(frameon=True, framealpha=0.75)
    plt.subplot(3,1,2)
    plt.plot(sync_frames(dynamic_tempo, duration_to_frame(y)), linewidth=1.5, label='tempo estimate')
    plt.legend(frameon=True, framealpha=0.75)
    plt.subplot(3,1,3)
    plt.plot(sync_frames(dynamic_tempo, duration_to_frame(y), np.std), linewidth=1.5, label='tempo estimate')
    plt.legend(frameon=True, framealpha=0.75)
    '''
    return dynamic_tempo


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


def remove_excess_labels(label, length):
    while len(label) > length:
        label = label[:-1]
    return label


# get directory for all the music
musicDir = get_directory()
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
i = 1103
for music in musicDir[1103:]:
    y, sr = librosa.load(path=music, sr=sr, offset=15)
    # get the music names
    music_name = os.path.basename(os.path.normpath(music)).split(".")[0]
    print("Extracting music feature: ", music_name, "...")

    # get all features
    print("extracting timbre...")
    mfcc_features = get_mfcc_feature(y)
    print("extracting melody...")
    chroma_features = get_chroma_feature(y)
    print("extracting energy...")
    rmse_features = get_rmse_feature(y)
    print("extracting rhythm...")
    rhythm_features = get_rhythm_feature(y)

    # get current feature labels
    row_arousal_label = arousal.loc[arousal['song_id'] == int(music_name)].index[0]
    row_valence_label = valence.loc[valence['song_id'] == int(music_name)].index[0]
    curr_arousal_label = arousal.values[row_arousal_label, 1:]
    curr_valence_label = valence.values[row_valence_label, 1:]
    # remove nan from list
    curr_arousal_label = curr_arousal_label[~np.isnan(curr_arousal_label)]
    curr_valence_label = curr_valence_label[~np.isnan(curr_valence_label)]
    # stack up all features
    all_feature = np.vstack([mfcc_features, chroma_features, rmse_features, rhythm_features])
    min_music_length = min(len(curr_arousal_label), len(curr_valence_label))
    # sync features to every 1 second
    # sync by using mean
    all_feature_m = sync_frames(all_feature, duration_to_frame(y, min_music_length))
    # sync by using standard deviation
    all_feature_std = sync_frames(all_feature, duration_to_frame(y, min_music_length), np.std)
    # remove excess value from both labels if any
    curr_arousal_label = remove_excess_labels(curr_arousal_label, min_music_length)
    curr_valence_label = remove_excess_labels(curr_valence_label, min_music_length)
    # append label to the data
    all_feature = np.vstack([all_feature_m, all_feature_std, curr_arousal_label, curr_valence_label])
    # save feature to csv
    with open("resource/features/" + music_name + ".csv", "w+", newline='') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerow(features_header)
        # invert feature to be (timestamp x feature) format
        csvWriter.writerows(np.transpose(all_feature))
    i += 1
    print(music_name, " done, total left: ", i, "/", len(musicDir))


'''
# plot for view
plt.figure(2, figsize=(12, 9))
plt.subplot(4, 1, 1)
librosa.display.specshow(chroma_features, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
plt.title('Chromagram')
plt.colorbar()

plt.subplot(4, 1, 2)
librosa.display.specshow(sync_frames(chroma_features, duration_to_frame(y)), sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
plt.title('Chromagram by frames')
plt.colorbar()

plt.subplot(4, 1, 3)
librosa.display.specshow(sync_frames(chroma_features, duration_to_frame(y), np.std), sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
plt.title('Chromagram by frames')
plt.colorbar()

plt.subplot(4, 1, 4)
librosa.display.specshow(sync_frames(chroma_features, duration_to_frame(y), np.median), sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
plt.title('Chromagram by frames')
plt.colorbar()
'''

plt.show()
