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
import traceback

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Task
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.app_constant import AlgorithmConstants, AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather


class ScatterAndGatherScaffold(ScatterAndGather):
    def __init__(
        self,
        min_clients: int = 1,
        num_rounds: int = 5,
        start_round: int = 0,
        wait_time_after_min_received: int = 10,
        aggregator_id=AppConstants.DEFAULT_AGGREGATOR_ID,
        aggregator_ctrl_id=AlgorithmConstants.SCAFFOLD_CTRL_AGGREGATOR_ID,
        persistor_id=AppConstants.DEFAULT_PERSISTOR_ID,
        shareable_generator_id=AppConstants.DEFAULT_SHAREABLE_GENERATOR_ID,
        train_task_name=AppConstants.TASK_TRAIN,
        train_timeout: int = 0,
        ignore_result_error: bool = True,
    ):
        """FederatedAveraging Workflow. The ScatterAndGatherScaffold workflow defines Federated training on all clients.
        The model persistor (persistor_id) is used to load the initial global model which is sent to all clients.
        Each clients sends it's updated weights after local training which is aggregated (aggregator_id). The
        shareable generator is used to convert the aggregated weights to shareable and shareable back to weights.
        The model_persistor also saves the model after training.

        Args:
            min_clients (int, optional): Min number of clients in training. Defaults to 1.
            num_rounds (int, optional): The total number of training rounds. Defaults to 5.
            start_round (int, optional): Start round for training. Defaults to 0.
            wait_time_after_min_received (int, optional): Time to wait before beginning aggregation after
                contributions received. Defaults to 10.
            train_timeout (int, optional): Time to wait for clients to do local training.
            aggregator_id (str, optional): ID of the aggregator component. Defaults to "aggregator".
            aggregator_ctrl_id (str, optional): ID of the aggregator component that aggregrates the SCAFFOLD control term. Defaults to "aggregator".
            persistor_id (str, optional): ID of the persistor component. Defaults to "persistor".
            shareable_generator_id (str, optional): ID of the shareable generator. Defaults to "shareable_generator".
            train_task_name (str, optional): Name of the train task. Defaults to "train".
        """

        super().__init__(
            min_clients=min_clients,
            num_rounds=num_rounds,
            start_round=start_round,
            wait_time_after_min_received=wait_time_after_min_received,
            aggregator_id=aggregator_id,
            persistor_id=persistor_id,
            shareable_generator_id=shareable_generator_id,
            train_task_name=train_task_name,
            train_timeout=train_timeout,
            ignore_result_error=ignore_result_error,
        )

        if not isinstance(aggregator_ctrl_id, str):
            raise TypeError("aggregator_ctrl_id must be a string.")

        # for SCAFFOLD
        self.aggregator_ctrl_id = aggregator_ctrl_id
        self.aggregator_ctrl = None
        self._global_ctrl_weights = None

    def start_controller(self, fl_ctx: FLContext) -> None:
        super().start_controller(fl_ctx=fl_ctx)

        # for SCAFFOLD
        if not self._global_weights:
            self.system_panic("Global weights not available!", fl_ctx)
            return

        self._global_ctrl_weights = copy.deepcopy(self._global_weights["weights"])
        # Initialize correction term with zeros
        for k in self._global_ctrl_weights.keys():
            self._global_ctrl_weights[k] = self._global_ctrl_weights[k] * 0
        # TODO: Print some stats of the correction magnitudes

        engine = fl_ctx.get_engine()
        self.aggregator_ctrl = engine.get_component(self.aggregator_ctrl_id)
        if not isinstance(self.aggregator_ctrl, Aggregator):
            self.system_panic(
                f"aggregator {self.aggregator_ctrl_id} must be an Aggregator type object but is "
                f"{type(self.aggregator_ctrl)}.",
                fl_ctx,
            )
            return

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        try:

            self.log_info(fl_ctx, "Beginning ScatterAndGatherScaffold training phase.")
            self._phase = AppConstants.PHASE_TRAIN

            fl_ctx.set_prop(AppConstants.PHASE, self._phase, private=True, sticky=False)
            fl_ctx.set_prop(AppConstants.NUM_ROUNDS, self._num_rounds, private=True, sticky=False)
            self.fire_event(AppEventType.TRAINING_STARTED, fl_ctx)

            for self._current_round in range(self._start_round, self._start_round + self._num_rounds):
                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.log_info(fl_ctx, f"Round {self._current_round} started.")
                fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self._global_weights, private=True, sticky=True)
                fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self._current_round, private=True, sticky=False)
                self.fire_event(AppEventType.ROUND_STARTED, fl_ctx)

                # Create train_task
                data_shareable: Shareable = self.shareable_gen.learnable_to_shareable(self._global_weights, fl_ctx)

                # add global SCAFFOLD controls
                # TODO: converting back and forth from Shareable doesn't seem ideal
                dxo = from_shareable(data_shareable)
                dxo.set_meta_prop(AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL, self._global_ctrl_weights)
                data_shareable = dxo.to_shareable()

                data_shareable.set_header(AppConstants.CURRENT_ROUND, self._current_round)
                data_shareable.set_header(AppConstants.NUM_ROUNDS, self._num_rounds)
                data_shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, self._current_round)

                train_task = Task(
                    name=self.train_task_name,
                    data=data_shareable,
                    props={},
                    timeout=self._train_timeout,
                    before_task_sent_cb=self._prepare_train_task_data,
                    result_received_cb=self._process_train_result,
                )

                self.broadcast_and_wait(
                    task=train_task,
                    min_responses=self._min_clients,
                    wait_time_after_min_received=self._wait_time_after_min_received,
                    fl_ctx=fl_ctx,
                    abort_signal=abort_signal,
                )

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.fire_event(AppEventType.BEFORE_AGGREGATION, fl_ctx)
                aggr_result = self.aggregator.aggregate(fl_ctx)
                ctrl_aggr_result = self.aggregator_ctrl.aggregate(fl_ctx)  # SCAFFOLD
                fl_ctx.set_prop(AppConstants.AGGREGATION_RESULT, aggr_result, private=True, sticky=False)
                self.fire_event(AppEventType.AFTER_AGGREGATION, fl_ctx)

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.fire_event(AppEventType.BEFORE_SHAREABLE_TO_LEARNABLE, fl_ctx)
                self._global_weights = self.shareable_gen.shareable_to_learnable(aggr_result, fl_ctx)

                # update SCAFFOLD global controls
                dxo = from_shareable(ctrl_aggr_result)
                ctr_diff = dxo.data
                for v_name, v_value in ctr_diff.items():
                    self._global_ctrl_weights[v_name] += v_value
                fl_ctx.set_prop(
                    AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL, self._global_ctrl_weights, private=True, sticky=True
                )

                fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self._global_weights, private=True, sticky=True)
                fl_ctx.sync_sticky()
                self.fire_event(AppEventType.AFTER_SHAREABLE_TO_LEARNABLE, fl_ctx)

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.fire_event(AppEventType.BEFORE_LEARNABLE_PERSIST, fl_ctx)
                self.persistor.save(self._global_weights, fl_ctx)
                self.fire_event(AppEventType.AFTER_LEARNABLE_PERSIST, fl_ctx)

                self.fire_event(AppEventType.ROUND_DONE, fl_ctx)
                self.log_info(fl_ctx, f"Round {self._current_round} finished.")

            self._phase = AppConstants.PHASE_FINISHED
            self.log_info(fl_ctx, "Finished ScatterAndGatherScaffold Training.")
        except BaseException as e:
            traceback.print_exc()
            error_msg = f"Exception in ScatterAndGatherScaffold control_flow: {e}"
            self.log_exception(fl_ctx, error_msg)
            self.system_panic(str(e), fl_ctx)

    def _accept_train_result(self, client_name: str, result: Shareable, fl_ctx: FLContext) -> bool:
        weights_accepted = super()._accept_train_result(client_name=client_name, result=result, fl_ctx=fl_ctx)

        # get SCAFFOLD updates
        dxo = from_shareable(result)
        c_delta_para = dxo.get_meta_prop(AlgorithmConstants.SCAFFOLD_CTRL_DIFF, None)

        # convert to Shareable for aggregation
        ctr_aggr_dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=c_delta_para)
        ctr_aggr_dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, dxo.get_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND))
        result["DXO"] = ctr_aggr_dxo._encode()  # TODO: uses protected function. Add way of replacing DXO in shareable

        # add SCAFFOLD control term
        ctrl_accepted = self.aggregator_ctrl.accept(result, fl_ctx)

        ctrl_accepted_msg = "ACCEPTED" if ctrl_accepted else "REJECTED"
        self.log_info(fl_ctx, f"Contribution from {client_name} {ctrl_accepted_msg} by the aggregator.")

        return weights_accepted and ctrl_accepted
