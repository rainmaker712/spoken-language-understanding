import torch
from torch import nn

class TokenNLLLoss(nn.Module):
    def __init__(self, ignore_index):
        super().__init__()

        self.nll_loss = nn.NLLLoss(reduction="sum", ignore_index=ignore_index)

    def forward(self, outputs, targets):
        batch_size, sequence_length = targets.size()
        outputs_flat = outputs.view(batch_size * sequence_length, -1)
        targets_flat = targets.view(-1)

        loss = self.nll_loss(outputs_flat, targets_flat)
        return loss


class TokenLabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, vocab_size, ignore_index=0):
        super(TokenLabelSmoothingLoss, self).__init__()

        assert 0.0 < label_smoothing <= 1.0

        self.kl_loss = nn.KLDivLoss(reduction="sum")

        smoothing_value = label_smoothing / (vocab_size - 2)  # -2 for pad and original value
        self.confidence = 1.0 - label_smoothing

        one_hot = torch.full(size=(vocab_size,), fill_value=smoothing_value)
        one_hot[ignore_index] = 0
        one_hot = one_hot.unsqueeze(0)
        self.register_buffer("one_hot", one_hot)

        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        """
        outputs (FloatTensor): batch_size, seq_length, n_classes
        targets (LongTensor): batch_size, seq_length
        """
        batch_size, sequence_length = targets.size()

        outputs_flat = outputs.view(batch_size * sequence_length, -1)
        targets_flat = targets.view(-1)
        ignore_positions = (targets_flat == self.ignore_index).unsqueeze(1)
        smoothed_targets = self.one_hot.repeat(len(targets_flat), 1)
        smoothed_targets.scatter_(dim=1, index=targets_flat.unsqueeze(1), value=self.confidence)
        smoothed_targets.masked_fill_(mask=ignore_positions, value=0)  # Make loss 0 for ignored positions
        loss = self.kl_loss(input=outputs_flat, target=smoothed_targets)

        return loss


class CTCLoss(nn.Module):
    def __init__(self, ignore_index):
        super().__init__()

        self.ctc_loss = nn.CTCLoss(blank=ignore_index, reduction="sum", zero_infinity=True)

    def forward(self, outputs, targets, memory_lengths, target_lengths):
        ctc_compatible_outputs = outputs.transpose(0, 1)
        loss = self.ctc_loss(
            log_probs=ctc_compatible_outputs,
            targets=targets,
            input_lengths=memory_lengths,
            target_lengths=target_lengths - 1,  # -1 for end token
        )

        return loss

def rought_accuracy_metric(targets, predictions, pad_id=0, end_id=3):
    """Calculate accuracy assuming the predictions length and the targets length are the same"""
    mask = (targets != pad_id) & (targets != end_id)
    correct = torch.as_tensor(targets[mask] == predictions[mask], dtype=torch.float)
    return correct.mean()