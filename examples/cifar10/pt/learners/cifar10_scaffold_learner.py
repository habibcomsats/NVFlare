# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from pt.learners.cifar10_learner import CIFAR10Learner

from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AlgorithmConstants


class CIFAR10ScaffoldLearner(CIFAR10Learner):
    """Simple Scaffold CIFAR-10 Trainer.
    Implements the training algorithm proposed in
    Karimireddy et al. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
    (https://arxiv.org/abs/1910.06378) using functions implemented in `ScaffoldLearner`.

    See `CIFAR10Learner` for Args and Returns.
    """

    def initialize(self, parts: dict, fl_ctx: FLContext):
        # Initialize super class and SCAFFOLD
        super().initialize(parts=parts, fl_ctx=fl_ctx)
        self.scaffold_init()

    def local_train(self, fl_ctx, train_loader, model_global, abort_signal: Signal, val_freq: int = 0):
        # local_train with SCAFFOLD steps
        c_global_para, c_local_para = self.scaffold_get_params()
        for epoch in range(self.aggregation_epochs):
            if abort_signal.triggered:
                return
            self.model.train()
            epoch_len = len(train_loader)
            self.epoch_global = self.epoch_of_start_time + epoch
            self.log_info(fl_ctx, f"Local epoch {self.client_id}: {epoch + 1}/{self.aggregation_epochs} (lr={self.lr})")

            for i, (inputs, labels) in enumerate(train_loader):
                if abort_signal.triggered:
                    return
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # FedProx loss term
                if self.fedproxloss_mu > 0:
                    fed_prox_loss = self.criterion_prox(self.model, model_global)
                    loss += fed_prox_loss

                loss.backward()
                self.optimizer.step()

                # SCAFFOLD step
                self.scaffold_model_update(c_global_para, c_local_para)

                current_step = epoch_len * self.epoch_global + i
                self.writer.add_scalar("train_loss", loss.item(), current_step)

            if val_freq > 0 and epoch % val_freq == 0:
                acc = self.local_valid(self.valid_loader, abort_signal, tb_id="val_acc_local_model", fl_ctx=fl_ctx)
                if acc > self.best_acc:
                    self.save_model(is_best=True)

        # Update the SCAFFOLD terms
        self.scaffold_terms_update(model_global, c_global_para, c_local_para)

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # return DXO with extra control differences for SCAFFOLD
        dxo = from_shareable(shareable)
        global_ctrl_weights = dxo.get_meta_prop(AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL)
        # convert to tensor and load into c_global model
        for k in global_ctrl_weights.keys():
            global_ctrl_weights[k] = torch.as_tensor(global_ctrl_weights[k])
        self.c_global.load_state_dict(global_ctrl_weights)

        # local training
        result_shareable = super().train(shareable, fl_ctx, abort_signal)
        if result_shareable.get_return_code() == ReturnCode.OK:
            # get DXO with weight updates from local training
            dxo = from_shareable(result_shareable)
            # add SCAFFOLD control
            if self.c_delta_para is None:
                raise ValueError("c_delta_para hasn't been computed yet!")
            dxo.set_meta_prop(AlgorithmConstants.SCAFFOLD_CTRL_DIFF, self.c_delta_para)

            return dxo.to_shareable()
        else:
            return result_shareable
