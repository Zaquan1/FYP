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

    def __init__(self, filepath, offset=0.0):
        self.filepath = filepath
        self.filename = os.path.basename(os.path.normpath(filepath)).split(".")[0]
        self.y, self.sr = librosa.load(path=filepath, sr=22050, offset=offset)
        self.frames = self.duration_to_frame()

        self.timbre_features = []
        self.melody_features = []
        self.energy_features = []
        self.rhythm_features = []

    def extract_timbre_features(self):
        print("extract timbre...")
        y_harmonic = librosa.effects.harmonic(self.y, margin=4)
        # get mfcc
        mfcc = librosa.feature.mfcc(y=y_harmonic, sr=self.sr, hop_length=self.hop_length, n_mfcc=13)
        # get dmfcc
        mfcc_delta = librosa.feature.delta(mfcc)
        # get ddmfcc
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        # get spectral flux
        spectral_flux = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        spectral_flux = spectral_flux / spectral_flux.max()
        # get zero crossing rate
        zc = librosa.feature.zero_crossing_rate(y=self.y)
        # get rolloff of 95% and 85%
        rolloff95 = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr, roll_percent=0.95)
        rolloff85 = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr, roll_percent=0.85)
        # get spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(self.y, sr=self.sr)
        # get spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(self.y, sr=self.sr)

        self.timbre_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2, spectral_flux,
                                          zc, rolloff95, rolloff85, spectral_contrast, spectral_centroid])
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

        return self.timbre_features

    def extract_melody_features(self):
        print("extract melody...")
        y_harmonic = librosa.effects.harmonic(self.y, margin=4)
        chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=self.sr, hop_length=self.hop_length, bins_per_octave=12 * 3)
        # get tonal centroid
        tonnets = librosa.feature.tonnetz(y_harmonic, sr=self.sr)

        self.melody_features = np.vstack([chromagram, tonnets])

        return self.melody_features

    def extract_energy_features(self):
        print("extract energy...")
        y_stft = librosa.stft(y=self.y, hop_length=self.hop_length)
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

        self.energy_features = np.vstack([rms, energy_novelty, loudness])
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
        return self.energy_features

    def extract_rhythm_features(self):
        print("extract rhythm...")
        onset = librosa.onset.onset_strength(self.y, sr=self.sr)
        dynamic_tempo = librosa.beat.tempo(onset_envelope=onset, sr=self.sr, aggregate=None)
        self.rhythm_features = dynamic_tempo
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
        return self.rhythm_features

    # get the frame for every sec_inc seconds
    def duration_to_frame(self, sec_inc=0.5):
        duration = librosa.get_duration(y=self.y, sr=self.sr, hop_length=self.hop_length)
        time_stamp = sec_inc
        time = []
        while time_stamp <= duration:
            time.append(time_stamp)
            time_stamp += sec_inc
        return librosa.time_to_frames(time, hop_length=self.hop_length, sr=self.sr)

    # sync all the frames based on duration_to_frame
    def sync_frames(self, features, aggregate=np.mean):
        sync_feature = librosa.util.sync(features, self.frames, aggregate=aggregate)
        return sync_feature

    def get_all_features(self):
        all_features = np.vstack([self.timbre_features, self.melody_features, self.energy_features,
                                  self.rhythm_features])
        return np.vstack([self.sync_frames(all_features), self.sync_frames(all_features, aggregate=np.std)])
