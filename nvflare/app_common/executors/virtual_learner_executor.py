# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import copy

from nvflare.apis.dxo import MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants, ValidateType
from nvflare.app_common.executors.learner_executor import LearnerExecutor
from nvflare.apis.fl_constant import ReservedKey
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator


class VirtualLearnerExecutor(LearnerExecutor):
    def __init__(
        self,
        learner_id,
        train_task=AppConstants.TASK_TRAIN,
        submit_model_task=AppConstants.TASK_SUBMIT_MODEL,
        validate_task=AppConstants.TASK_VALIDATION,
        virtual_clients = 1,
        real_clients = 8,
        use_local_aggregation = True
    ):
        """Key component to run learner on clients.

        Args:
            learner_id (str): id pointing to the learner object
            train_task (str, optional): label to dispatch train task. Defaults to AppConstants.TASK_TRAIN.
            submit_model_task (str, optional): label to dispatch submit model task. Defaults to AppConstants.TASK_SUBMIT_MODEL.
            validate_task (str, optional): label to dispatch validation task. Defaults to AppConstants.TASK_VALIDATION.
        """
        super().__init__(
            learner_id=learner_id,
            train_task=train_task,
            submit_model_task=submit_model_task,
            validate_task=validate_task,
        )
        self.virtual_clients = virtual_clients
        self.real_clients = real_clients
        self.real_id_name = ""  # real client ID
        self.use_local_aggregation = use_local_aggregation
        if self.use_local_aggregation:
            self.aggregator = InTimeAccumulateWeightedAggregator()
        else:
            self.aggregator = None

    def initialize(self, fl_ctx: FLContext):
        super().initialize(fl_ctx=fl_ctx)
        self.real_id_name = fl_ctx.get_identity_name()
        self.log_info(fl_ctx, f"Simulating {self.virtual_clients} virtual client(s) on client {self.real_id_name}.")

    #def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
    #    super().execute(task_name=task_name, shareable=shareable, fl_ctx=fl_ctx, abort_signal=abort_signal)

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_debug(fl_ctx, f"train abort signal: {abort_signal.triggered}")

        shareable.set_header(AppConstants.VALIDATE_TYPE, ValidateType.BEFORE_TRAIN_VALIDATE)
        validate_result: Shareable = self.learner.validate(shareable, fl_ctx, abort_signal)

        train_result = self._train_virtual_clients(shareable, fl_ctx, abort_signal)
        print("@1 train_result", type(train_result))
        if not (train_result and isinstance(train_result, Shareable)):
            print("@2 ReturnCode.EMPTY_RESULT", ReturnCode.EMPTY_RESULT)
            return make_reply(ReturnCode.EMPTY_RESULT)

        # if the learner returned the valid BEFORE_TRAIN_VALIDATE result, set the INITIAL_METRICS in
        # the train result, which can be used for best model selection.
        if (
            validate_result
            and isinstance(validate_result, Shareable)
            and validate_result.get_return_code() == ReturnCode.OK
        ):
            try:
                metrics_dxo = from_shareable(validate_result)
                train_dxo = from_shareable(train_result)
                train_dxo.meta[MetaKey.INITIAL_METRICS] = metrics_dxo.data.get(MetaKey.INITIAL_METRICS, 0)
                print("@Return train_dxo", type(train_dxo))
                _result = train_dxo.to_shareable()
                print("@Return _result", type(_result))
                return _result
            except ValueError:
                print("@3 train_result", type(train_result))
                return train_result
        else:
            print("@4 train_result", type(train_result))
            return train_result

    def _train_virtual_clients(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:

        train_results = {}
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        print("############## current_round", current_round)

        # loop through virtual clients
        for vid in range(1, self.virtual_clients+1):
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            # TODO: more general client identity management
            real_id = int(self.real_id_name.replace("site-", ""))
            assert real_id > 0, f"expecting real_ids starting from 1 but got {real_id}"
            #if self.virtual_clients > 1:
            virtual_id = (real_id-1)*self.virtual_clients + vid
            #else:
            #    virtual_id = real_id  # TODO: there must be a better way
            print(f"DEBUG: real ID {real_id}, local virtual client ID {vid}, virtual ID {virtual_id}")

            virtual_name = f"site-{virtual_id}"
            self.log_info(fl_ctx, f"Training on virtual client {virtual_name}")
            fl_ctx.set_prop(AppConstants.VIRTUAL_NAME, virtual_name)

            # reinitialize only to begin training
            engine = fl_ctx.get_engine()
            self.learner.initialize(engine.get_all_components(), fl_ctx)
            _train_result = self.learner.train(shareable=copy.deepcopy(shareable), fl_ctx=fl_ctx, abort_signal=abort_signal)  # TODO: is deepcopy needed?
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            # Set information for local aggregation
            _train_result.set_peer_props({ReservedKey.IDENTITY_NAME: virtual_name})
            _train_result.set_header(AppConstants.CONTRIBUTION_ROUND, current_round)
            fl_ctx.set_prop(AppConstants.CURRENT_ROUND, current_round)
            print("@@@@@@@ current_round", current_round)
            print("@@@@@@@ _train_result.get_header(AppConstants.CONTRIBUTION_ROUND)", _train_result.get_header(AppConstants.CONTRIBUTION_ROUND))

            # Append to train_results or add to aggregator
            if self.aggregator:
                print("###### accept aggregate")
                self.aggregator.accept(shareable=_train_result, fl_ctx=fl_ctx)
            else:
                # add DXO to collection
                train_results.update({virtual_id: from_shareable(_train_result)})

        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        if self.aggregator:
            self.log_info(fl_ctx, f"Compute local aggregate of {self.virtual_clients} virtual clients.")
            result = self.aggregator.aggregate(fl_ctx=fl_ctx)
            print("########## result", type(result))
            return result
        else:
            raise NotImplemented("Needs testing!")
            return DXO(data_kind=DataKind.COLLECTION, data=train_results).to_shareable()


    def finalize(self, fl_ctx: FLContext):
        try:
            if self.learner:
                self.learner.finalize(fl_ctx)
        except Exception as e:
            self.log_exception(fl_ctx, f"learner finalize exception: {e}")
