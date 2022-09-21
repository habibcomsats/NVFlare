# Copyright (c) 2021-2022, [BLINDED] CORPORATION.  All rights reserved.
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

from typing import Dict, List

from flare.apis.client import Client
from flare.apis.fl_context import FLContext
from flare.apis.impl.controller import ClientTask
from flare.apis.shareable import Shareable
from flare.apis.signal import Signal
from flare.app_common.workflows.statistics_controller import StatisticsController


class MockStatisticsController(StatisticsController):
    def __init__(self, metric_configs: Dict[str, dict], writer_id: str):
        super(MockStatisticsController, self).__init__(metric_configs, writer_id)

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        pass

    def start_controller(self, fl_ctx: FLContext):
        pass

    def stop_controller(self, fl_ctx: FLContext):
        pass

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        pass

    def metrics_task_flow(self, abort_signal: Signal, fl_ctx: FLContext, metric_task: str):
        pass

    def results_cb(self, client_task: ClientTask, fl_ctx: FLContext):
        pass

    def post_fn(self, task_name: str, fl_ctx: FLContext):
        pass
