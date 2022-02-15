import torch
import torch.nn as nn
import random
from models.transformer import SinusoidalPositionalEncoding, TransformerEncoder, TransformerDecoder
from models.transformer import mask_from_subsequent_positions, mask_from_lengths


class GlobalAttention(nn.Module):
    def __init__(self):
        super(GlobalAttention, self).__init__()

        self.softmax = nn.Softmax(dim=2)

    def forward(self, memory, decoder_outputs, mask=None):
        # memory: (batch_size, source_sequence_length, hidden_size)
        # decoder_outputs: (batch_size, target_sequence_length, hidden_size)

        scores = torch.bmm(decoder_outputs, memory.transpose(1, 2))
        # scores: (batch_size, target_sequence_length, source_sequence_length)

        if mask is not None:
            scores.masked_fill_(mask, -float("inf"))

        attention = self.softmax(scores)

        return attention

class AttentionPoolingLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.layer = nn.Linear(self.input_dim, 1, bias=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, inputs, batch_len):
        # input size (B, enc_T, input_dim)

        if inputs.size(1) != torch.max(batch_len):
            batch_len = torch.tensor([x + 1 for x in batch_len])

        # define Mask for attention
        mask = []
        max_len = torch.max(batch_len)  # 배치 내 최대길이
        for _len in batch_len:
            # if audio length is smaller than max
            if max_len - _len > 0:
                mask += [
                    torch.cat(
                        [torch.zeros(1, _len), torch.ones(1, (max_len - _len))], dim=-1
                    )
                ]
            else:
                mask += [torch.zeros(1, _len)]
        mask = torch.cat(mask, dim=0).unsqueeze(1).bool().to("cuda")
        mask = mask.squeeze(dim=1)

        # set mask
        self.set_mask(mask)

        ## 어텐션 스코어 계산 score = w2 * tanh ( w1 * H' ), (B, enc_T)
        score = (self.layer(inputs)).squeeze(dim=-1)

        # Masking
        if self.mask is not None:
            score.data.masked_fill_(self.mask, -float("inf"))

        # compute attention weight, (B, enc_T)
        # a = softmax ( w2 * tanh ( w1 * H' ) )
        attn_weights = self.softmax(score)

        # weighted sum = context vector (B, 1, attn_dim)
        context = torch.bmm(attn_weights.unsqueeze(dim=1), inputs)

        return context.squeeze(dim=1), attn_weights

class TransformerASREncoder(nn.Module):
    def __init__(self, in_features, num_layers, d_model, d_ff, n_heads, dropout):
        super(TransformerASREncoder, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(in_features, d_model),
            SinusoidalPositionalEncoding(encoding_size=d_model, initial_length=1024),
            nn.Dropout(dropout),
        )

        self.transformer_encoder = TransformerEncoder(
            num_layers=num_layers, d_model=d_model, d_ff=d_ff, n_heads=n_heads, dropout=dropout
        )

        self.encoder_output_size = d_model

    def forward(self, features, feature_lengths):
        # features: (batch_size, subsampled_sequence_length, subsampled_feature_size)

        length_mask = mask_from_lengths(lengths=feature_lengths, max_length=features.size(1))

        x = self.input_layer(features)
        outputs = self.transformer_encoder(x, sources_key_padding_mask=length_mask)

        return outputs, feature_lengths


class TransformerASRDecoder(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, d_ff, n_heads, dropout, pad_id):
        super(TransformerASRDecoder, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Embedding(vocab_size, d_model, padding_idx=pad_id),
            SinusoidalPositionalEncoding(encoding_size=d_model, initial_length=512),
            nn.Dropout(dropout),
        )

        self.transformer_decoder = TransformerDecoder(
            num_layers=num_layers, d_model=d_model, d_ff=d_ff, n_heads=n_heads, dropout=dropout, use_memory=True
        )
        self.output_layer = nn.Sequential(nn.Linear(d_model, vocab_size), nn.LogSoftmax(dim=2))

        self.decoder_output_size = vocab_size  # For ctc layer

    def forward_step(self, inputs, input_lengths, memory, memory_lengths, state):

        inputs_subsequent_mask = mask_from_subsequent_positions(size=inputs.size(1)).to(inputs.device)
        memory_lengths_mask = mask_from_lengths(lengths=memory_lengths, max_length=memory.size(1))

        if input_lengths is not None:
            input_lengths_mask = mask_from_lengths(lengths=input_lengths, max_length=inputs.size(1))
        else:
            input_lengths_mask = None

        x = self.input_layer(inputs)

        x, (state, memory_attention) = self.transformer_decoder(
            x,
            memory=memory,
            inputs_mask=inputs_subsequent_mask,
            inputs_key_padding_mask=input_lengths_mask,
            memory_key_padding_mask=memory_lengths_mask,
            state=state,
        )

        outputs = self.output_layer(x)
        attention = torch.mean(memory_attention, dim=1).detach()

        return outputs, (state, attention)

    def forward(
        self, inputs=None, input_lengths=None, memory=None, memory_lengths=None, state=None, decoding_sampling_rate=1.0,
    ):

        decoding_sampling_rate = 0.0

        assert memory is not None or self.attention_module is None
        assert inputs is not None or decoding_sampling_rate == 1.0

        sample_previous_token = True if random.random() < decoding_sampling_rate else False

        if not sample_previous_token:

            outputs, (state, _) = self.forward_step(
                inputs=inputs, input_lengths=input_lengths, memory=memory, memory_lengths=memory_lengths, state=state
            )
        else:
            output_list = []
            inputs_step = inputs[:, 0].unsqueeze(1)
            sequence_length = inputs.size(1)

            for i in range(sequence_length):
                output_step, (state, _) = self.forward_step(
                    inputs=inputs_step, input_lengths=None, memory=memory, memory_lengths=memory_lengths, state=state
                )

                max_values, max_indices = output_step.max(dim=-1)
                inputs_step = max_indices
                output_list.append(output_step)

            outputs = torch.cat(output_list, dim=1).contiguous()

        return outputs, state


class SquareCNNSubsampler(nn.Module):
    def __init__(self, in_features, out_features, num_layers):
        """
        from https://github.com/espnet/espnet/blob/9ec6939a580c4f8ab6cefb9cb13614e44e72a627/espnet/nets/pytorch_backend/transformer/subsampling.py#L14
        """
        super(SquareCNNSubsampler, self).__init__()

        if num_layers == 2:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1, out_channels=out_features, kernel_size=3, stride=2),
                nn.BatchNorm2d(num_features=out_features),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=2),
                nn.BatchNorm2d(num_features=out_features),
                torch.nn.ReLU(),
            )
            self.out_features = out_features * (((in_features - 1) // 2 - 1) // 2)
        elif num_layers == 1:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1, out_channels=out_features, kernel_size=3, stride=2),
                nn.BatchNorm2d(num_features=out_features),
                torch.nn.ReLU(),
            )

            self.out_features = out_features * ((in_features - 1) // 2)
        else:
            raise NotImplementedError

        self.num_layers = num_layers

    def forward(self, features, feature_lengths):
        # features: (batch_size, sequence_length, feature_size)

        x = features.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        
        if self.num_layers == 2:
            downsampled_feature_lengths = ((feature_lengths - 1) // 2 - 1) // 2
        elif self.num_layers == 1:
            downsampled_feature_lengths = (feature_lengths - 1) // 2
        else:
            raise NotImplementedError

        return x, downsampled_feature_lengths

from torch import nn


class VGGSubsampler(nn.Module):
    def __init__(self, in_features):
        super(VGGSubsampler, self).__init__()

        self.vgg = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # Half sequence length
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # Half sequence length
        )

        self.out_features = (in_features // 4) * 128

        self.bridge_batch_norm = nn.BatchNorm1d(num_features=self.out_features)

    def forward(self, features, feature_lengths):
        # features: (batch_size, sequence_length, feature_size, 3)


        features = features.transpose(3, 1)
        # (batch_size, sequence_length, feature_size, 3) -> (batch_size, 3, feature_size, sequence_length)

        x = self.vgg(features)
        # x: (batch_size, 128, feature_size / 4, sequence_length / 4)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = self.bridge_batch_norm(x)
        x = x.transpose(1, 2)
        x = x.contiguous()
        # x: (batch_size, sequence_length / 4, 128 * feature_size / 4)

        subsampled_feature_lengths = feature_lengths // 4

        return x, subsampled_feature_lengths

class Classifier(nn.Module):
    def __init__(self, hidden_size, num_intent):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(hidden_size, num_intent)
        self.linear.weight.data.normal_(mean=0.0, std=0.02)
        self.linear.bias.data.zero_()

    def forward(self, out):
        out = self.dropout(out)
        out = self.linear(out)
        return out