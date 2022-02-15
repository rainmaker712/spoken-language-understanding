from pathlib import Path
from acoustics.mrcg import mrcg
import librosa
import numpy as np
import soundfile
import torch
from joblib import Memory
from scipy import signal


class SpectralFeatureExtractor:
    def __init__(
        self,
        sample_rate=16000,
        window_ms=30,
        hop_ms=10,
        speech_enhancement=None,
        remove_silence=False,
        silence_threshold=30,
        feature_type="log-mel",
        n_fft=512,
        n_mels=80,
        n_mfcc=26,
        mrcg_bands=64,
        delta=False,
        delta_delta=False,
        stack=False,
        cmvn=False,
        transform_fn=None,
        cache_location="./cachedir",
    ):
        if feature_type is None and cache_location is not None:
            raise ValueError("Cache is not allowed when using raw waveform feature")
        if feature_type is None and hop_ms != 10:
            raise ValueError("Raw feature only support hop_ms = 10")

        self.cache_location = cache_location
        self.memory = Memory(self.cache_location, verbose=0)
        self.extract_static_feature = self.memory.cache(self.extract_static_feature)
        self.sample_rate = sample_rate
        self.window_samples = int(window_ms / 1000 * sample_rate)
        self.hop_samples = int(hop_ms / 1000 * sample_rate)
        self.speech_enhancement = speech_enhancement
        self.remove_silence = remove_silence
        self.silence_threshold = silence_threshold
        self.feature_type = feature_type
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.mrcg_bands = mrcg_bands
        self.delta = delta
        self.delta_delta = delta_delta
        self.stack = stack
        self.cmvn = cmvn
        self.transform_fn = transform_fn

        if feature_type == "spectrogram":
            feature_size = n_fft / 2 + 1
        elif feature_type in ("mel", "log-mel"):
            feature_size = n_mels
        elif feature_type == "mfcc":
            feature_size = n_mfcc
        elif feature_type == "mrcg":
            feature_size = 256
        elif feature_type is None:
            feature_size = 1
        else:
            raise NotImplementedError

        if delta and delta_delta:
            multiple = 3
        elif delta:
            multiple = 2
        else:
            multiple = 1

        if stack:
            bin_size = feature_size * multiple
        else:
            feature_size *= multiple
            bin_size = feature_size

        self.feature_size = feature_size
        self.bin_size = bin_size

    def __call__(self, audio_path: Path, augment=False, cache=True):
        # feature: (feature_size, time_size)

        if self.feature_type is None:
            audio, audio_sample_rate = self.open_audio_file(audio_path=str(audio_path))
            if self.remove_silence:
                audio = self.remove_silence_from_audio(audio=audio, silence_threshold=self.silence_threshold)
            return audio

        if cache:
            feature = self.extract_static_feature(
                # audio_path=str(audio_path.resolve()),
                audio_path=audio_path,
                sample_rate=self.sample_rate,
                window_samples=self.window_samples,
                hop_samples=self.hop_samples,
                speech_enhancement=self.speech_enhancement,
                remove_silence=self.remove_silence,
                silence_threshold=self.silence_threshold,
                feature_type=self.feature_type,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                n_mfcc=self.n_mfcc,
                mrcg_bands=self.mrcg_bands,
            )
        else:
            feature = self.extract_static_feature.func(
                # audio_path=str(audio_path.resolve()),
                audio_path=audio_path,
                sample_rate=self.sample_rate,
                window_samples=self.window_samples,
                hop_samples=self.hop_samples,
                speech_enhancement=self.speech_enhancement,
                remove_silence=self.remove_silence,
                silence_threshold=self.silence_threshold,
                feature_type=self.feature_type,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                n_mfcc=self.n_mfcc,
                mrcg_bands=self.mrcg_bands,
            )

        if augment:
            assert self.transform_fn is not None
            feature = self.transform_fn(feature).squeeze()

        if self.cmvn:
            feature = (feature - np.expand_dims(feature.mean(axis=1), axis=1)) / np.expand_dims(
                (feature.std(axis=1) + 1e-16), axis=1
            )

        feature_list = [feature]
        if self.delta:
            feature_delta = librosa.feature.delta(feature, axis=1, width=9)
            feature_list.append(feature_delta)

        if self.delta_delta:
            feature_delta_delta = librosa.feature.delta(feature, axis=1, width=9, order=2)
            feature_list.append(feature_delta_delta)

        if self.stack:
            full_feature = np.stack(feature_list, axis=2)
        else:
            full_feature = np.concatenate(feature_list, axis=0)

        # full_feature: (feature_size, time_size, [stack_size]) -> (time_size, feature_size, [stack_size])
        full_feature = np.swapaxes(full_feature, 0, 1)

        return full_feature

    @staticmethod
    def extract_static_feature(
        audio_path,
        sample_rate,
        window_samples,
        hop_samples,
        remove_silence,
        silence_threshold,
        speech_enhancement,
        feature_type,
        n_fft,
        n_mels,
        n_mfcc,
        mrcg_bands,
    ):

        audio, audio_sample_rate = SpectralFeatureExtractor.open_audio_file(audio_path=audio_path)

        assert (
            audio_sample_rate == sample_rate
        ), f"Sample rate of file {audio_sample_rate} is not equal to {sample_rate}"

        if speech_enhancement:
            audio = SpectralFeatureExtractor.enhance_speech(
                audio, algorithm=speech_enhancement, sample_rate=audio_sample_rate
            )

        if remove_silence:
            audio = SpectralFeatureExtractor.remove_silence_from_audio(audio=audio, silence_threshold=silence_threshold)

        if feature_type == "spectrogram":
            stft = torch.stft(
                torch.FloatTensor(audio),
                n_fft,
                hop_length=hop_samples,
                win_length=window_samples,
                window=torch.hamming_window(window_samples),
                center=False,
                normalized=False,
                onesided=True,
            )

            stft = (stft[:, :, 0].pow(2) + stft[:, :, 1].pow(2)).pow(0.5)
            feature = stft.numpy()
        elif feature_type == "mel":
            feature = librosa.feature.melspectrogram(
                y=audio, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_samples, win_length=window_samples,
            )
        elif feature_type == "log-mel":
            feature = librosa.feature.melspectrogram(
                y=audio, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_samples, win_length=window_samples,
            )
            feature = np.log(feature + 1e-6)
        elif feature_type == "mfcc":
            feature = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=n_mfcc,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_samples,
                win_length=window_samples,
            )
            # feat[0] = librosa.feature.rmse(y, hop_samples=st, frame_length=ws)
        elif feature_type == "mrcg":
            feature = mrcg(
                y=audio,
                sample_rate=sample_rate,
                frame1_length=window_samples,
                hop_length=hop_samples,
                num_bands=mrcg_bands,
            )
            feature = np.swapaxes(feature, 0, 1)  # (time, spectrum) -> (spectrum -> time)
        else:
            raise ValueError(f"Unsupported Acoustic Feature: {feature_type}")

        return feature

    @staticmethod
    def open_audio_file(audio_path: str):
        # if audio_path.rsplit(".", maxsplit=1)[1] == "pcm":
        #     with open(audio_path, "rb") as audio_file:
        #         # Convert 16 bit signed (-32768, +32767) to float
        #         audio = np.fromfile(audio_file, dtype=np.int16).astype(np.single) / 32768
        #         audio_sample_rate = 16000
        # else:
        audio, audio_sample_rate = soundfile.read(audio_path, dtype=np.single)

        target_sample_rate = 16000

        if audio_sample_rate != target_sample_rate:

            # if sample rate is not 8k, resample as 16k
            # y = librosa.resample(y, sr, target_sample_rate)

            secs = len(audio) / audio_sample_rate
            n_samples = int(secs * target_sample_rate)
            audio = signal.resample(audio, n_samples)

        return audio, target_sample_rate

    @staticmethod
    def enhance_speech(audio, algorithm, sample_rate):

        return audio

    @staticmethod
    def remove_silence_from_audio(audio, silence_threshold):
        non_silence_indices = librosa.effects.split(audio, top_db=silence_threshold)
        audio = np.concatenate([audio[start:end] for start, end in non_silence_indices])
        return audio