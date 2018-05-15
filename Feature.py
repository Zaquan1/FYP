import librosa
import librosa.display
import numpy as np
import os


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
        return self.timbre_features

    def extract_melody_features(self):
        print("extract melody...")
        y_harmonic = librosa.effects.harmonic(self.y, margin=4)
        # get chromagram
        chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
                                                sr=self.sr,
                                                hop_length=self.hop_length,
                                                bins_per_octave=12 * 3)
        # get tonal centroid
        tonnets = librosa.feature.tonnetz(y_harmonic, sr=self.sr)

        self.melody_features = np.vstack([chromagram, tonnets])
        return self.melody_features

    def extract_energy_features(self):
        print("extract energy...")
        y_stft = librosa.stft(y=self.y, hop_length=self.hop_length)
        S, phase = librosa.magphase(y_stft)
        # get rms
        rms = librosa.feature.rmse(S=S).flatten()
        rms_diff = np.zeros_like(rms)
        rms_diff[1:] = np.diff(rms)
        # get energy novelty
        energy_novelty = np.max([np.zeros_like(rms_diff), rms_diff], axis=0)
        # extracting loudness
        power = np.abs(S)**2
        p_mean = np.sum(power, axis=0, keepdims=True)
        p_ref = np.max(power)
        # get loudness
        loudness = librosa.power_to_db(p_mean, ref=p_ref)

        self.energy_features = np.vstack([rms, energy_novelty, loudness])
        return self.energy_features

    def extract_rhythm_features(self):
        print("extract rhythm...")
        onset = librosa.onset.onset_strength(self.y, sr=self.sr)
        # get dynamic tempo
        dynamic_tempo = librosa.beat.tempo(onset_envelope=onset, sr=self.sr, aggregate=None)
        self.rhythm_features = dynamic_tempo

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
