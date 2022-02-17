import os
import torch
from torch import nn
from models.layers import Classifier, AttentionPoolingLayer, TransformerASRDecoder, TransformerASREncoder, SquareCNNSubsampler
from transformers import RobertaModel, RobertaTokenizer
from utils.asr_utils import CTCLoss, TokenLabelSmoothingLoss, rough_accracy_metric

class SLU(nn.Module):

    def __init__(self, args, hparams):
        super(SLU, self).__init__()
        
        hparams = hparams.transformer_model

        # Set Config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Model
        self.subsampler = SquareCNNSubsampler(
            in_features=args.feature_size,
            out_features=hparams["subsampler-size"],
            num_layers=hparams.get("square-cnn-num-layers", 1),
        )
        
        self.transformer_encoder = TransformerASREncoder(
                in_features=self.subsampler.out_features,
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

    def forward(self, input_batch, mode='train'):                

        features = input_batch["audio_features"]
        feature_lengths = input_batch["audio_lengths"]
        intent_labels = input_batch["intents"]

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


class Seq2SeqASR(nn.Module):

    def __init__(self, args, hparams):
        super(Seq2SeqASR, self).__init__()
        
        hparams = hparams.transformer_model

        # Use ASR Pretrained Model
        self.tokenizer = RobertaTokenizer.from_pretrained(args.text_model_name)
        
        self.pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
        self.sos_id = self.tokenizer.convert_tokens_to_ids("[CLS]")
        self.end_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
        self.vocab_size = len(self.tokenizer)

        # Set Config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load ASR Model
        self.subsampler = SquareCNNSubsampler(
            in_features=args.feature_size,
            out_features=hparams["subsampler-size"],
            num_layers=hparams.get("square-cnn-num-layers", 1),
        )
        
        self.transformer_encoder = TransformerASREncoder(
                in_features=self.subsampler.out_features,
                num_layers=hparams["num-encoder-layers"],
                d_model=hparams["transformer-d-model"],
                d_ff=hparams["transformer-d-ff"],
                n_heads=hparams["transformer-n-heads"],
                dropout=hparams["dropout"],
            )

        self.transformer_decoder = TransformerASRDecoder(
                vocab_size = self.vocab_size,
                num_layers=hparams["num-encoder-layers"],
                d_model=hparams["transformer-d-model"],
                d_ff=hparams["transformer-d-ff"],
                n_heads=hparams["transformer-n-heads"],
                dropout=hparams["dropout"],
                pad_id=self.pad_id,
            )

        # ASR Loss
        if args.enable_ctc:
            self.ctc_layer = nn.Sequential(
                nn.Linear(self.transformer_encoder.encoder_output_size, self.transformer_decoder.decoder_output_size),
                nn.LogSoftmax(dim=-1)
            )
            self.ctc_loss_function = CTCLoss(ignore_index=self.pad_id)
        else:
            self.ctc_layer = None
            self.ctc_loss_function = None

        self.seq2seq_loss = TokenLabelSmoothingLoss(
            label_smoothing=0.1, ignore_index=self.pad_id, vocab_size=self.vocab_size
        )

        self.ctc_loss_weights = args.ctc_loss_weights

    def forward(self, input_batch, mode='train'):                

        features = input_batch["audio_features"]
        feature_lengths = input_batch["audio_lengths"]
        text_inputs = input_batch["text_inputs"]
        text_lengths = input_batch["text_lengths"]
        text_target_inputs = input_batch["text_target_inputs"]
        text_target_lengths = input_batch["text_target_lengths"]

        memory, memory_lengths = self.encode(features, feature_lengths)
        
        ctc_outputs = self.ctc(memory)
        
        decoder_outputs, _ = self.decoder(
            inputs=text_inputs,
            input_lengths=text_lengths,
            memroy=memory,
            memory_lengths=memory_lengths,
            decoding_sampling_rate=self.decoding_sampling_rate
        )

        max_val, predictions = decoder_outputs.max(dim=-1)
        seq2seq_loss = self.seq2seq_loss(outputs=decoder_outputs, targets=text_target_inputs)
        ctc_loss = self.ctc_loss_function(
            outputs = ctc_outputs,
            targets=text_target_inputs,
            memory_lengths=memory_lengths,
            target_lengths=text_target_lengths
        )

        loss = (1 - self.ctc_loss_weights) * seq2seq_loss + self.ctc_loss_weights * ctc_loss
        loss_scale = text_target_inputs.sum().item()

        scaled_loss = loss / loss_scale
        scaled_loss = scaled_loss.unsqueeze(dim=0)

        asr_acc = rough_accracy_metric(
            targets=text_target_inputs,
            predictions=predictions,
            pad_id=self.pad_id,
            end_id=self.end_id
        )

        asr_acc = asr_acc.unsqueeze(dim=0).cpu()

        outputs = {
            "loss": scaled_loss,
            "accuracy": asr_acc,
        }

        return outputs

    def encode(self, features, feature_lengths):

        subsampled_features, subsampled_feature_lengths = self.subsampler(features, feature_lengths)
        memory, memory_lengths = self.transformer_encoder(subsampled_features, subsampled_feature_lengths)

        return memory, memory_lengths

    def ctc(self, memory):
        if self.ctc_layer is not None:
            ctc_outputs =self.ctc_layer(memory)
        else:
            ctc_outputs = None
        return ctc_outputs