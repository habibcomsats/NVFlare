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

from abc import ABC, abstractmethod

from flare.apis.fl_component import FLComponent
from flare.apis.fl_context import FLContext
from flare.app_common.abstract.learnable import Learnable


class LearnablePersistor(FLComponent, ABC):
    @abstractmethod
    def load(self, fl_ctx: FLContext) -> Learnable:
        """Load the Learnable object.

        Args:
            fl_ctx: FLContext

        Returns:
            Learnable object loaded

        """
        pass

    @abstractmethod
    def save(self, learnable: Learnable, fl_ctx: FLContext):
        """Persist the Learnable object.

        Args:
            learnable: the Learnable object to be saved
            fl_ctx: FLContext

        """
        pass
