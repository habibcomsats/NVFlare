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

# This SCAFFOLD implementation is based on https://github.com/Xtra-Computing/NIID-Bench

# MIT License
#
# Copyright (c) 2021 Yiqun Diao, Qinbin Li
#
# Copyright (c) 2020 International Business Machines
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import copy

import torch
from pt.learners.cifar10_learner import CIFAR10Learner
from torch.optim import Optimizer

from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AlgorithmConstants, AppConstants


class CIFAR10ScaffoldLearner(CIFAR10Learner):
    def __init__(
        self,
        dataset_root: str = "./dataset",
        aggregation_epochs: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
        lr: float = 1e-2,
        fedproxloss_mu: float = 0.0,
        central: bool = False,
        analytic_sender_id: str = "analytic_sender",
    ):
        """Simple Scaffold CIFAR-10 Trainer.
        Implements the training algorithm proposed in
        Karimireddy et al. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
        (https://arxiv.org/abs/1910.06378)
        This SCAFFOLD implementation is based on https://github.com/Xtra-Computing/NIID-Bench

        Args:
            dataset_root: directory with CIFAR-10 data.
            aggregation_epochs: the number of training epochs for a round. Defaults to 1.
            train_task_name: name of the task to train the model.
            submit_model_task_name: name of the task to submit the best local model.
            analytic_sender_id: id of `AnalyticsSender` if configured as a client component. If configured, TensorBoard events will be fired. Defaults to "analytic_sender".

        Returns:
            a Shareable with the updated local model after running `execute()`
            or the best local model depending on the specified task.
        """
        super().__init__(
            dataset_root=dataset_root,
            aggregation_epochs=aggregation_epochs,
            train_task_name=train_task_name,
            submit_model_task_name=submit_model_task_name,
            lr=lr,
            fedproxloss_mu=fedproxloss_mu,
            central=central,
            analytic_sender_id=analytic_sender_id,
        )

        # SCAFFOLD control terms
        self.cnt = 0
        self.curr_lr = None
        self.c_global = None
        self.c_local = None
        self.c_delta_para = None

    def scaffold_init(self):
        # create models for SCAFFOLD correction terms
        self.c_global = copy.deepcopy(self.model)
        self.c_local = copy.deepcopy(self.model)
        # Initialize correction term with zeros
        c_init_para = self.model.state_dict()
        for k in c_init_para.keys():
            c_init_para[k] = c_init_para[k] * 0
        self.c_global.load_state_dict(c_init_para)
        self.c_local.load_state_dict(c_init_para)

    def scaffold_model_update(self, c_global_para, c_local_para):
        # Update model using scaffold controls
        # See https://github.com/Xtra-Computing/NIID-Bench/blob/main/experiments.py#L391
        self.curr_lr = self._get_lr_values(self.optimizer)[0]
        net_para = self.model.state_dict()
        for key in net_para:
            net_para[key] = net_para[key] - self.curr_lr * (c_global_para[key] - c_local_para[key])
        self.model.load_state_dict(net_para)

        self.cnt += 1

    def scaffold_terms_update(self, model_global, c_global_para, c_local_para):
        # Update the local scaffold controls
        # See https://github.com/Xtra-Computing/NIID-Bench/blob/main/experiments.py#L403

        c_new_para = self.c_local.state_dict()
        self.c_delta_para = copy.deepcopy(self.c_local.state_dict())
        global_model_para = model_global.state_dict()
        net_para = self.model.state_dict()
        for key in net_para:
            c_new_para[key] = (
                c_new_para[key]
                - c_global_para[key]
                + (global_model_para[key] - net_para[key]) / (self.cnt * self.curr_lr)
            )
            self.c_delta_para[key] = (c_new_para[key] - c_local_para[key]).cpu().numpy()
        self.c_local.load_state_dict(c_new_para)

    def _get_lr_values(self, optimizer: Optimizer):
        """
        This function is used to get the learning rates of the optimizer.
        """
        return [group["lr"] for group in optimizer.state_dict()["param_groups"]]

    def initialize(self, parts: dict, fl_ctx: FLContext):
        super().initialize(parts=parts, fl_ctx=fl_ctx)
        self.scaffold_init()

    def local_train(self, fl_ctx, train_loader, model_global, abort_signal: Signal, val_freq: int = 0):
        self.cnt = 0
        # Adapted from https://github.com/Xtra-Computing/NIID-Bench/blob/main/experiments.py#L371
        c_global_para = self.c_global.state_dict()
        c_local_para = self.c_local.state_dict()
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
