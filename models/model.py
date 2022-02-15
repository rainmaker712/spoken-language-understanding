import os
import torch
from torch import nn
from models.layers import Classifier, AttentionPoolingLayer, TransformerASREncoder, SquareCNNSubsampler, VGGSubsampler
from transformers import RobertaModel, RobertaTokenizer

class SLU(nn.Module):

    def __init__(self, args, hparams):
        super(SLU, self).__init__()
        
        hparams = hparams.transformer_model

        # Set Config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load ASR Model
        # self.subsampler = VGGSubsampler(
        #     in_features=feature_size
        # )

        self.subsampler = SquareCNNSubsampler(
            in_features=args.feature_size,
            out_features=hparams["subsampler-size"],
            num_layers=hparams.get("square-cnn-num-layers", 1),
        )
        
        self.transformer_encoder = TransformerASREncoder(
                in_features=hparams["subsampler-size"],
                num_layers=hparams["num-encoder-layers"],
                d_model=hparams["transformer-d-model"],
                d_ff=hparams["transformer-d-ff"],
                n_heads=hparams["transformer-n-heads"],
                dropout=hparams["dropout"],
            )

        
        # text model
        # self.tokenizer = RobertaTokenizer.from_pretrained(args.text_model_name)
        # self.text_model = RobertaModel.from_pretrained(args.text_model_name)

        # For SLU Model
        self.pooling_layer = AttentionPoolingLayer(args.output_embedding_size)
        self.classifier = Classifier(hidden_size=args.output_embedding_size, num_intent=args.num_intents)

    def forward(self, features=None, feature_lengths=None, intent_labels = None, mode='train'):

        result = {}

        memory, memory_lengths = self.encode(features, feature_lengths)
        output, attn_weights = self.pooling_layer(memory, memory_lengths)
        intent_logits = self.classifier(output)
        value, predicted_intent = intent_logits.max(1)

        intent_loss = torch.nn.functional.cross_entropy(intent_logits, intent_labels)
        intent_acc = torch.eq(predicted_intent, intent_labels).float().sum() / intent_labels.size(0)

        result['loss'] = intent_loss
        result['accuracy'] = intent_acc

        if mode=='val':
            result['pred_intent'] = predicted_intent
            result['predicted_scores'] = torch.topk(intent_logits, 3)[0]
            result['predicted_intents'] = torch.topk(intent_logits, 3)[1]

        return result

    def encode(self, features, feature_lengths):

        subsampled_features, subsampled_feature_lengths = self.subsampler(features, feature_lengths)
        memory, memory_lengths = self.transformer_encoder(subsampled_features, subsampled_feature_lengths)

        return memory, memory_lengths