import os
import copy
import torch
import logging
from pytz import timezone
from datetime import datetime
from omegaconf import OmegaConf

from utils.data_loader import fsc_dataloader
from optimizer.optimizer import build_optimizer
from trainer.slu_trainer import Trainer
from utils.arguments import get_slu_args
from utils.utils import init_report_dict, set_random_seed

# SLU
from models.model import SLU

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = int(torch.cuda.device_count())

if __name__ == "__main__":

    args = get_slu_args()
    hparams = OmegaConf.load("config/config.yaml")

    gpu_available = True if n_gpu > 0 else False
    
    #set seeds
    set_random_seed(seed=1234, gpu=gpu_available)


    logger.info(f"***** model init *****")
    model = SLU(args, hparams)
    model.to(device)

    # Build datalodaer
    logger.info(f"***** preparing training files *****")
    
    # load text model    
    train_loader, valid_loader, test_loader = fsc_dataloader(base_path=args.data_path, batch_size=args.batch_size, num_workers=args.num_workers, text_model_name=args.text_model_name)

    # Build Optimizer
    logger.info(f"***** building optimizer *****")
    num_step_per_epoch = len(train_loader)
    total_train_steps = num_step_per_epoch * args.epochs

    optimizer, scheduler = build_optimizer(args, model, total_train_steps,
                                       logger, use_scheduler=args.use_scheduler)
    
    # Build trainer
    logger.info(f"***** training start *****")

    report_dict = init_report_dict(['loss', 'accuracy'])
    trainer = Trainer(
                      model=model,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      data_loader=train_loader,
                      num_step_per_epoch=num_step_per_epoch,
                      total_train_steps=total_train_steps,
                      report_dict = report_dict,
                      config=args,
                      n_gpu=n_gpu,
                      logger=logger,
                      )

    best_valid_acc = 0
    best_test_acc = 0
    best_report = None
    best_test_report = None
    global_step, epoch = 0, 0

    for epoch in range(args.epochs):
        logger.info("==================== Epoch %d of %d ====================" % (epoch, args.epochs))

        global_step = trainer.train(global_step)
        records = trainer.eval(valid_loader, epoch, global_step)

        if best_report is None or records['accuracy'] > best_valid_acc:

            logger.info(f"New Records for SLU Valid Data: {best_valid_acc} -> {records['accuracy']} at epoch {epoch}")

            best_valid_epoch = epoch
            best_valid_acc = records['accuracy']
            best_report = copy.copy(records)

            test_records = trainer.eval(test_loader, epoch, global_step, data_type='test')

            if best_test_report is None or test_records["accuracy"] > best_test_acc:
                logger.info(f"New Records for SLU Test Data: {best_test_acc} -> {test_records['accuracy']}  at epoch {epoch}")

                best_epoch = epoch
                best_test_acc = test_records['accuracy']
                best_test_report = copy.copy(records)

        best_metric_str = '[So far best validation]  epoch:  %d  ' % (best_valid_epoch)
        best_metric_str += '  '.join(f"{metric}: {v:.5f}" for metric, v in best_report.items())

        best_metric_str = '[So far best test]  epoch:  %d  ' % (best_epoch)
        best_metric_str += '  '.join(f"{metric}: {v:.5f}" for metric, v in test_records.items())

        loss = test_records["loss"]
        f1 = test_records["macro_f1"]
        acc = test_records["accuracy"]

        logger.info(best_metric_str)

    best_epoch = epoch