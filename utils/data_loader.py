import os
import re

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn import preprocessing

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from acoustics.spectral_feature_extractor import SpectralFeatureExtractor
from transformers import RobertaTokenizer

conf = OmegaConf.load("config/config.yaml")
hparams = conf.feature_extractor

class BaseDataset(Dataset):

    def load_audio(self, idx):
        df_row = self.df.iloc[idx]
        filename = os.path.join(self.base_path, df_row['path'])
        audio_feature = self.audio_extractor(filename, cache=True)
        intent = df_row['intent_label']
        transcript = df_row['transcription']
        text_encoding = self.text_tokenizer(transcript, padding=True, truncation=True, return_tensors="pt")['input_ids']
        
        return audio_feature, intent, text_encoding, transcript

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raise NotImplementedError

    def labels_list(self):
        raise NotImplementedError

    def audio_extractor(self, audio_path, cache=False):

        feature_extractor = SpectralFeatureExtractor(
            sample_rate=hparams["audio-sample-rate"],
            window_ms=hparams["window-ms"],
            hop_ms=hparams["hop-ms"],
            speech_enhancement=hparams.get("speech-enhancement"),
            remove_silence=hparams["remove-silence"],
            feature_type=hparams["feature"],
            n_fft=hparams["n-fft"],
            n_mels=hparams["n-mels"],
            n_mfcc=hparams["n-mfcc"],
            cache_location=hparams["cache-dir"],
        )

        return feature_extractor(audio_path, cache=cache)



class FluentSpeechDataset(BaseDataset):
    
    def __init__(self, base_path, split="train", intent_encoder=None, text_model_name=None):
        
        self.base_path = base_path
        self.text_tokenizer = RobertaTokenizer.from_pretrained(text_model_name)
        self.df = pd.read_csv(os.path.join(base_path, "data/", '{}_data.csv'.format(split)))
        self.df['intent'] = self.df[['action', 'object', 'location']].apply('_'.join, axis=1)
        
        if intent_encoder is None:
            intent_encoder = preprocessing.LabelEncoder()
            intent_encoder.fit(self.df['intent'])
        self.intent_encoder = intent_encoder
        self.df['intent_label'] = intent_encoder.transform(self.df['intent'])

        self.labels_set = set(self.df['intent_label'])
        self.label2idx = {}
        for label in self.labels_set:
            idx = np.where(self.df['intent_label'] == label)[0]
            self.label2idx[label] = idx
        
    def labels_list(self):
        return self.intent_encoder.classes_

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_feature, intent, text_encoding, trasncription = self.load_audio(idx)

        return audio_feature, intent, text_encoding, trasncription

def fsc_collate_classifier(inputs):
    
    # audio_feature, audio_length, intents, text_encodings, transcriptions = zip(*inputs)
    max_frame_length = 813  # 99% percent, librosa - logmel
    inputs = sorted(inputs, key=lambda sample: sample[0].shape[0], reverse=True)

    wav_inputs, wav_lengths = [], []
    intents = []
    
    for batch in inputs:

        wav_input, intent, _, _ = batch
        wav_input = wav_input[:max_frame_length]
        wav_inputs.append(torch.tensor(wav_input))
        wav_lengths.append(torch.tensor(len(wav_input)))
        intents.append(torch.tensor(intent).long())

    wav_inputs = pad_sequence(wav_inputs, batch_first=True)
    wav_lengths = torch.stack(wav_lengths)
    intents = torch.stack(intents)

    data_out = {
        "audio_features": wav_inputs,
        "audio_lengths": wav_lengths,
        "intents": intents
    }

    return data_out

def fsc_dataloader(base_path, batch_size, num_workers=0, text_model_name=None):
    
    train_dataset = FluentSpeechDataset(base_path, split="train", text_model_name=text_model_name)
    valid_dataset = FluentSpeechDataset(base_path, split="valid", 
                        intent_encoder=train_dataset.intent_encoder, text_model_name=text_model_name)
    test_dataset = FluentSpeechDataset(base_path, split="test", 
                        intent_encoder=train_dataset.intent_encoder, text_model_name=text_model_name)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=fsc_collate_classifier, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=fsc_collate_classifier, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=fsc_collate_classifier, num_workers=num_workers)

    return train_loader, val_loader, test_loader