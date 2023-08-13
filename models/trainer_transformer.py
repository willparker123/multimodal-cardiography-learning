# -*- coding: utf-8 -*-

#!/usr/bin/env python3
import os
import subprocess
from collections import defaultdict
from helpers import create_new_folder

import numpy as np
import torch
import textwrap
import logging
from torch import nn
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import config
from typing import Any, List, Optional, Sequence, Tuple, Union, Dict
import time
from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.cfg import CFG
from .scoring_metrics import evaluate_scores
import torch.nn.functional as nnf

from torch_ecg.cfg import CFG, DEFAULTS
from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn
from torch.utils.data.dataset import Dataset
import math
from config import outputpath
from dataset import ECGPCGDataset

TrainCfg = CFG()
TrainCfg.classes = ["N", "A"]
# configs of training epochs, batch, etc.
TrainCfg.n_epochs = config.global_opts.epochs
# TODO: automatic adjust batch size according to GPU capacity
# https://stackoverflow.com/questions/45132809/how-to-select-batch-size-automatically-to-fit-gpu
TrainCfg.batch_size = config.global_opts.batch_size

# configs of optimizers and lr_schedulers
TrainCfg.optimizer = "adamw_amsgrad"  # "sgd", "adam", "adamw"
TrainCfg.momentum = 0.949  # default values for corresponding PyTorch optimizers
TrainCfg.betas = (0.9, 0.999)  # default values for corresponding PyTorch optimizers
TrainCfg.decay = 1e-2  # default values for corresponding PyTorch optimizers

TrainCfg.learning_rate = 1e-4  # 1e-3
TrainCfg.lr = TrainCfg.learning_rate

TrainCfg.lr_scheduler = "one_cycle"  # "one_cycle", "plateau", "burn_in", "step", None
TrainCfg.lr_step_size = 50
TrainCfg.lr_gamma = 0.1
TrainCfg.max_lr = 2e-3  # for "one_cycle" scheduler, to adjust via expriments

TrainCfg.burn_in = 400
TrainCfg.steps = [5000, 10000]

TrainCfg.early_stopping = CFG()  # early stopping according to challenge metric
TrainCfg.early_stopping.min_delta = 0.001  # should be non-negative
TrainCfg.early_stopping.patience = 10

# configs of loss function
# TrainCfg.loss = "BCEWithLogitsLoss"
# TrainCfg.loss = "BCEWithLogitsWithClassWeightLoss"
TrainCfg.loss = "AsymmetricLoss"  # "FocalLoss"
TrainCfg.loss_kw = CFG(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp")
TrainCfg.flooding_level = (
    0.0  # flooding performed if positive, typically 0.45-0.55 for cinc2021?
)

TrainCfg.monitor = "challenge_metric"

TrainCfg.log_step = 20
TrainCfg.eval_every = 20

# configs of model selection
# "resnet_nature_comm_se", "multi_scopic_leadwise", "vgg16", "vgg16_leadwise",
TrainCfg.cnn_name = "transformer"
TrainCfg.rnn_name = "none"  # "none", "lstm"
TrainCfg.attn_name = "none"  # "none", "se", "gc", "nl"

# configs of inputs and outputs
# almost all records have duration >= 8s, most have duration >= 10s
# use `utils.utils_signal.ensure_siglen` to ensure signal length
TrainCfg.input_len = int(2000 * 8.0)
# tolerance for records with length shorter than `TrainCfg.input_len`
TrainCfg.input_len_tol = int(0.2 * TrainCfg.input_len)
TrainCfg.sig_slice_tol = 0.4  # None, do no slicing
TrainCfg.siglen = TrainCfg.input_len


# extends BaseTrainer from torch_ecg; based on CINC2021Trainer
class TransformerTrainer(BaseTrainer):
    """ """

    __name__ = "TransformerTrainer"

    def __init__(
        self,
        model: nn.Module,
        model_config: dict,
        train_config: dict,
        device: Optional[torch.device] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        model: Module,
            the model to be trained
        model_config: dict,
            the configuration of the model,
            used to keep a record in the checkpoints
        train_config: dict,
            the configuration of the training,
            including configurations for the data loader, for the optimization, etc.
            will also be recorded in the checkpoints.
            `train_config` should at least contain the following keys:
                "monitor": str,
                "loss": str,
                "n_epochs": int,
                "batch_size": int,
                "learning_rate": float,
                "lr_scheduler": str,
                    "lr_step_size": int, optional, depending on the scheduler
                    "lr_gamma": float, optional, depending on the scheduler
                    "max_lr": float, optional, depending on the scheduler
                "optimizer": str,
                    "decay": float, optional, depending on the optimizer
                    "momentum": float, optional, depending on the optimizer
        device: torch.device, optional,
            the device to be used for training
        lazy: bool, default True,
            whether to initialize the data loader lazily
        """
        train_config.physionetOnly = kwargs.get('physionetOnly',True)
        super().__init__(
            model=model,
            dataset_cls=ECGPCGDataset,
            model_config=model_config,
            train_config=train_config,
            device=device,
            lazy=lazy,
        )

    def _setup_dataloaders(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
    ) -> None:
        """
        setup the dataloaders for training and validation
        Parameters
        ----------
        train_dataset: Dataset, optional,
            the training dataset
        val_dataset: Dataset, optional,
            the validation dataset
        """
        dataset = self.dataset_cls(#config=self.train_config, training=True, lazy=False
                                    clip_length=config.global_opts.segment_length, 
                                    ecg_sample_rate=config.global_opts.sample_rate_ecg,
                                    pcg_sample_rate=config.global_opts.sample_rate_pcg,
                                    verifyComplete=False)
        if self.train_config.physionetOnly is not None and self.train_config.physionetOnly:
            dataset = self.dataset_cls(clip_length=config.global_opts.segment_length, 
                                        ecg_sample_rate=config.global_opts.sample_rate_ecg,
                                        pcg_sample_rate=config.global_opts.sample_rate_pcg,
                                        verifyComplete=False,
                                        paths_ecgs=[outputpath+f'physionet/data_ecg_{config.global_opts.ecg_type}/'], 
                                        paths_pcgs=[outputpath+f'physionet/data_pcg_{config.global_opts.pcg_type}/'], 
                                        paths_csv=[outputpath+f'physionet/data_physionet_raw'])
        
        train_len = math.floor(len(dataset)*config.global_opts.train_split)
        data_train, data_test = torch.utils.data.random_split(dataset, [train_len, len(dataset)-train_len], generator=torch.Generator().manual_seed(42)) 
        if train_dataset is None:
            data_train
        
        if self.train_config.debug:
            val_train_dataset = data_train
        else:
            val_train_dataset = None
        if val_dataset is None:
            val_dataset = data_test
        print(f"len data_train, data_test: {len(data_train)} {len(data_test)}")
        print(f"len val_train_dataset, val_dataset: {len(val_train_dataset)} {len(val_dataset)}")

        # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/4
        num_workers = config.global_opts.number_of_processes

        self.train_loader = DataLoader(
            dataset=data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        #self.test_loader = DataLoader(
        #    dataset=data_test,
        #    batch_size=self.batch_size,
        #    shuffle=True,
        #    num_workers=num_workers,
        #    pin_memory=True,
        #    drop_last=False,
        #    collate_fn=collate_fn,
        #)
        if self.train_config.debug:
            self.val_train_loader = DataLoader(
                dataset=val_train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
            )
        else:
            self.val_train_loader = None
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    def run_one_step(
        self, *data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        data: tuple of Tensors,
            the data to be processed for training one step (batch),
            should be of the following order:
            signals, labels, *extra_tensors
        Returns
        -------
        preds: Tensor,
            the predictions of the model for the given data
        labels: Tensor,
            the labels of the given data
        """
        self.model.train()
        signals, labels = data
        signals = signals.to(self.device)
        labels = labels.to(self.device)
        preds = self.model(signals)
        return preds, labels

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """ """
        self.model.eval()
        all_scalar_preds = []
        all_bin_preds = []
        all_labels = []

        for signals, labels in data_loader:
            signals = signals.to(device=self.device, dtype=self.dtype)
            labels = labels.numpy()
            all_labels.append(labels)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Forward pass, get our logits
            criterion = nn.CrossEntropyLoss()
            logits = self.model(signals)
            labels = np.array(list(map(lambda x: 0 if x == -1 else (1 if x == 1 else x), labels)))
            print(f"logits {np.shape(logits)} {logits}")
            print(f"labels {np.shape(torch.from_numpy(labels))} {torch.from_numpy(labels.squeeze())}")
            # Calculate the loss with the logits and the labels
            loss = criterion(logits, torch.from_numpy(labels.squeeze()))
            prob = nnf.softmax(logits, dim=1)
            top_p, top_class = prob.topk(1, dim = 1)
            pred = top_class
            all_scalar_preds.append(prob)
            all_bin_preds.append(pred)

        all_scalar_preds = np.concatenate(all_scalar_preds, axis=0)
        all_bin_preds = np.concatenate(all_bin_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        classes = self.model.classes

        if self.val_train_loader is not None:
            msg = f"all_scalar_preds.shape = {all_scalar_preds.shape}, all_labels.shape = {all_labels.shape}"
            self.log_manager.log_message(msg, level=logging.DEBUG)
            head_num = len(classes)
            head_scalar_preds = all_scalar_preds[:head_num, ...]
            head_bin_preds = all_bin_preds[:head_num, ...]
            print(f"head_bin_preds {head_bin_preds} {np.shape(head_bin_preds)}")
            print(f"np.where(row) {[np.array(head_bin_preds)[row] for row in range(len(head_bin_preds))]}")
            head_preds_classes = [
                np.array(head_bin_preds)[row] for row in range(len(head_bin_preds))
            ]
            head_labels = all_labels[:head_num, ...]
            head_labels_classes = [
                np.array(head_labels)[row] for row in range(len(head_labels))
            ]
            for n in range(head_num):
                msg = textwrap.dedent(
                    f"""
                ----------------------------------------------
                scalar prediction:    {[round(n, 3) for n in head_scalar_preds[n].tolist()]}
                binary prediction:    {head_bin_preds[n].tolist()}
                labels:               {head_labels[n].astype(int).tolist()}
                predicted classes:    {head_preds_classes[n].tolist()}
                label classes:        {head_labels_classes[n].tolist()}
                ----------------------------------------------
                """
                )
                self.log_manager.log_message(msg)

        (
            auroc,
            auprc,
            accuracy,
            f_measure,
            f_beta_measure,
            g_beta_measure,
            challenge_metric,
        ) = evaluate_scores(
            classes=classes,
            truth=all_labels,
            scalar_pred=all_scalar_preds,
            binary_pred=all_bin_preds,
        )
        eval_res = dict(
            auroc=auroc,
            auprc=auprc,
            accuracy=accuracy,
            f_measure=f_measure,
            f_beta_measure=f_beta_measure,
            g_beta_measure=g_beta_measure,
            challenge_metric=challenge_metric,
        )

        # in case possible memeory leakage?
        del all_scalar_preds, all_bin_preds, all_labels

        return eval_res
        #self.model.eval()

    @property
    def batch_dim(self) -> int:
        """
        batch dimension, for CinC2021, it is 0,
        """
        return 0

    @property
    def extra_required_train_config_fields(self) -> List[str]:
        """ """
        return []

    @property
    def save_prefix(self) -> str:
        return f"{self._model.__name__}_{self.model_config.cnn.name}_epoch"

    def extra_log_suffix(self) -> str:
        return super().extra_log_suffix() + f"_{self.model_config.cnn.name}"
    
def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)

