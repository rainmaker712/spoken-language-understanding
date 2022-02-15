import os
import re

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn import preprocessing

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch._six import container_abcs, string_classes, int_classes
from acoustics.spectral_feature_extractor import SpectralFeatureExtractor
from transformers import RobertaTokenizer

conf = OmegaConf.load("config/config.yaml")
hparams = conf.feature_extractor

class BaseDataset(Dataset):

    def load_audio(self, idx):
        df_row = self.df.iloc[idx]
        filename = os.path.join(self.base_path, df_row['path'])
        # audio_feature, sample_rate = torchaudio.load(filename)
        # audio_feature = audio_feature.squeeze()
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
    collate = variable_length_collate(variable_length_fields={"features"})
    inputs = sorted(inputs, key=lambda sample: sample[0].shape[0], reverse=True)

    wav_inputs, wav_lengths = [], []
    intents = []
    
    for batch in inputs:

        wav_input, intent, _, _ = batch
        wav_input = wav_input[:max_frame_length]
        wav_inputs.append({"features": wav_input})
        wav_lengths.append(torch.tensor(len(wav_input)))
        intents.append(torch.tensor(intent).long())

    wav_inputs = collate(batch=wav_inputs)
    wav_inputs = wav_inputs["features"]
    wav_lengths = torch.stack(wav_lengths)
    intents = torch.stack(intents)

    # audio_feature = collate(audio_feature)
    # collate = variable_length_collate()
    # intents = torch.LongTensor(intents)
    # audio_feature = rnn.pad_sequence(audio_feature, batch_first=True)
    # text_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # text_encodings = text_tokenizer(transcriptions, padding=True, truncation=True, return_tensors="pt")['input_ids']
    # text_encodings = rnn.pad_sequence(text_encodings, batch_first=True)

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


np_str_obj_array_pattern = re.compile(r"[SaUO]")
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, " "dicts or lists; found {}"
)

def variable_length_collate(variable_length_fields=None):

    # Adapted from pytorch default_collate
    def collate(batch, variable_length=False):
        r"""Puts each data field into a tensor with outer dimension batch size"""

        elem = batch[0]
        elem_type = type(elem)
        if variable_length:
            return pad_sequence([torch.as_tensor(d) for d in batch], batch_first=True, padding_value=0)
        elif isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == "numpy" and elem_type.__name__ != "str_" and elem_type.__name__ != "string_":
            elem = batch[0]
            if elem_type.__name__ == "ndarray":
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))

                return collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {
                key: collate([d[key] for d in batch], variable_length=key in variable_length_fields) for key in elem
            }
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
            return elem_type(*(collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            transposed = zip(*batch)
            return [collate(samples) for samples in transposed]

        raise TypeError(default_collate_err_msg_format.format(elem_type))

    return collate
