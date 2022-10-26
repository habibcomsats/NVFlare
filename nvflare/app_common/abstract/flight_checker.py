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

from abc import ABC, abstractmethod

from nvflare.apis.client import Client
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class FlightChecker(FLComponent, ABC):
    @abstractmethod
    def create_task_data(self,
                         task_name: str,
                         fl_ctx: FLContext) -> Shareable:
        """Create the task data for the check request to clients
        This method is called at the beginning of the FlightCheck controller.
        The internal state of the checker should be reset here, if the checker is used multiple times.

        Args:
            task_name: name of the task
            fl_ctx: FL context

        Returns: task data as a shareable
        """
        pass

    @abstractmethod
    def process_client_report(
            self,
            client: Client,
            task_name: str,
            report: Shareable,
            fl_ctx: FLContext) -> bool:
        """Accept the shareable submitted by a client.
        This method is called every time a report is received from a client.

        Args:
            client: the client that submitted report
            task_name: name of the task that the report corresponds to
            report: client submitted report
            fl_ctx: FLContext

        Returns:
            boolean to indicate if the client data is acceptable.
            If not acceptable, the control flow will exit.

        """
        pass

    @abstractmethod
    def final_check(self, fl_ctx: FLContext) -> bool:
        """Perform the final check.
        This method is called after received reports from all clients.

        Args:
            fl_ctx: FLContext

        Returns:
            boolean indicating whether the flight check is successful.
            If not successful, the control flow will exit.
        """
        pass

