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


XGB_TRAIN_TASK = "train"


class XGBShareableHeader:
    WORLD_SIZE = "_world_size"
    RANK_MAP = "_rank_map"
    XGB_FL_SERVER_PORT = "_server_port"
    XGB_FL_SERVER_SECURE = "_secure_server"
