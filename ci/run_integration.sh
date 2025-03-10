#!/bin/bash
#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
#

# Argument(s):
#   BUILD_TYPE:   numpy | tensorflow | pytorch | overseer | ha | auth, tests to execute

set -ex
BUILD_TYPE=pytorch

if [[ $# -eq 1 ]]; then
    BUILD_TYPE=$1

elif [[ $# -gt 1 ]]; then
    echo "ERROR: too many parameters are provided"
    exit 1
fi

init_pipenv() {
    echo "initializing pip environment: $1"
    pipenv install -r "$1"
    export PYTHONPATH=$PWD
}

remove_pipenv() {
    echo "removing pip environment"
    pipenv --rm
    rm Pipfile Pipfile.lock
}

integration_test_pt() {
    echo "Run PT integration test..."
    init_pipenv requirements-dev.txt
    testFolder="tests/integration_test"
    clean_up_snapshot_and_job
    pushd ${testFolder}
    pipenv run ./run_integration_tests.sh -m pytorch
    popd
    clean_up_snapshot_and_job
    remove_pipenv
}

integration_test_tf() {
    echo "Run TF integration test..."
    # not using pipenv because we need tensorflow package from the container
    python -m pip install -r requirements-dev.txt
    export PYTHONPATH=$PWD
    testFolder="tests/integration_test"
    clean_up_snapshot_and_job
    pushd ${testFolder}
    ./run_integration_tests.sh -m tensorflow
    popd
    clean_up_snapshot_and_job
}

add_dns_entries() {
    echo "adding DNS entries for HA test cases"
    cp /etc/hosts /etc/hosts_bak
    echo "127.0.0.1 localhost0 localhost1" | tee -a /etc/hosts > /dev/null
}

remove_dns_entries() {
    echo "restoring original /etc/hosts file"
    cp /etc/hosts_bak /etc/hosts
}

clean_up_snapshot_and_job() {
    rm -rf /tmp/nvflare*
}

integration_test() {
    echo "Run integration test..."
    init_pipenv requirements-dev.txt
    add_dns_entries
    testFolder="tests/integration_test"
    clean_up_snapshot_and_job
    pushd ${testFolder}
    pipenv run ./run_integration_tests.sh -m "$1"
    popd
    clean_up_snapshot_and_job
    remove_dns_entries
    remove_pipenv
}

case $BUILD_TYPE in

    tensorflow)
        echo "Run TF tests..."
        integration_test_tf
        ;;

    pytorch)
        echo "Run PT tests..."
        integration_test_pt
        ;;

    ha)
        echo "Run HA tests..."
        integration_test "ha"
        ;;

    numpy)
        echo "Run numpy tests..."
        integration_test "numpy"
        ;;

    overseer)
        echo "Run overseer tests..."
        integration_test "overseer"
        ;;

    auth)
        echo "Run auth tests..."
        integration_test "auth"
        ;;

    *)
        echo "ERROR: unknown parameter: $BUILD_TYPE"
        ;;
esac
