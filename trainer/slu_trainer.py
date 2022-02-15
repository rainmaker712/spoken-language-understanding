import os
import copy

import torch
import numpy as np
from sklearn.metrics import f1_score

from utils.utils import TrainReport

class Trainer(object):

    def __init__(self, model, optimizer, scheduler, data_loader,
                 num_step_per_epoch, total_train_steps, report_dict,
                 config, n_gpu, logger):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_step_per_epoch = num_step_per_epoch
        self.total_train_steps = total_train_steps
        self.data_loader = data_loader

        self.report_dict = report_dict
        self.n_gpu = n_gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger

        self.config = config

    def save_checkpoint(self, global_step, model, config, best_report, model_nm):
        save_args = {
            'config': config,
            'step': global_step,
            'model': model.state_dict(),
            'report': best_report
        }

        model_nm = model_nm + ".pth"
        torch.save(save_args, os.path.join('checkpoint', model_nm))
        self.logger.info(f"**** cslu save model at {global_step} ****")

    def train(self, global_step):
        report = TrainReport(self.report_dict)
        report.reset_dict()

        self.model.train()
        self.logger.info("**** Train slu data ****")

        # for test
        num_examples = 0
        for idx, wav_batch in enumerate(self.data_loader):
            report, num_examples, global_step = self.wav_step(wav_batch, report, num_examples,global_step)

            if global_step % 100 == 0:
                self.print_update_report(report, num_examples, global_step)
                report.reset_dict()
                num_examples = 0

            if global_step % 2000 == 0
                temp_model_nm = "temp_cslu"
                self.save_checkpoint(global_step, self.model, None, None, temp_model_nm)

        report.reset_dict()

        return global_step

    def wav_step(self, wav_batch, report, num_examples, global_step):

        wav_batch["audio_features"] = wav_batch["audio_features"].to(self.device)
        wav_batch["audio_lengths"] = wav_batch["audio_lengths"].to(self.device)
        wav_batch["intents"] = wav_batch["intents"].to(self.device)

        result = self.model(wav_batch)

        batch_size = wav_batch["audio_lengths"].size(0)
        report, num_examples = self.loss_update(result, batch_size, report, num_examples, global_step)
        global_step += 1

        del result

        return report, num_examples, global_step

    def loss_update(self, result, batch_size, report, num_examples, global_step):

        loss = result['loss']

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        report.append_batch_result(batch_size, result)
        num_examples += batch_size

        del loss

        return report, num_examples

    def print_update_report(self, report, num_examples, global_step):
        report.compute_mean(num_examples)
        prefix_str = "[%d/%d]   " % (global_step, self.total_train_steps)
        report_str = report.result_str()
        self.logger.info(prefix_str + report_str)

        records = copy.copy(report.__dict__['report_dict'])
        if self.scheduler is not None:
            lr_this_step = self.scheduler.get_lr()[0]
            records['lr'] = lr_this_step

    def eval(self, test_loader, epoch, global_step, data_type='val'):

        self.logger.info("*********** start evaluation **************")

        self.model.eval()
        report = TrainReport(self.report_dict)
        report.reset_dict()

        predicted_intent = np.empty(0)
        target_intent = np.empty(0)
        for idx, wav_batch in enumerate(test_loader):
            with torch.no_grad():

                wav_batch["audio_features"] = wav_batch["audio_features"].to(self.device)
                wav_batch["audio_lengths"] = wav_batch["audio_lengths"].to(self.device)
                wav_batch["intents"] = wav_batch["intents"].to(self.device)

                result = self.model(wav_batch, mode='val')
                batch_size = wav_batch["audio_lengths"].size(0)
                
                num_examples += batch_size
                report.append_batch_result(batch_size, result)

                predicted_intent = np.append(predicted_intent, result['pred_intent'].cpu().numpy(), axis=0)
                target_intent = np.append(target_intent, wav_batch["intent_labels"].cpu().numpy(), axis=0)

        report.compute_mean(num_examples)
        macro_f1 = f1_score(target_intent, predicted_intent, average='macro')

        records = copy.copy(report.__dict__['report_dict'])
        report.compute_mean(num_examples)
        records['macro_f1'] = macro_f1
        if data_type == 'val':
            prefix_str = "[Validation]   "
        else:
            prefix_str = "[Test]   "
        report_str = '   '.join(f"{metric}: {v:.5f}" for metric, v in records.items())
        self.logger.info(prefix_str + report_str)

        return records