# Copyright (c) 2022, [BLINDED] CORPORATION.  All rights reserved.
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

import argparse
import os
import signal

import docker
from flare.lighter.utils import generate_password


def start(port, folder, dashboard_image, env_vars, passphrase):
    if not folder:
        folder = os.getcwd()
    environment = dict()
    if env_vars:
        for e in env_vars:
            splitted = e.split("=")
            environment[splitted[0]] = splitted[1]
    if passphrase:
        environment["FL_DASHBOARD_PP"] = passphrase
    if not os.path.exists(os.path.join(folder, "db.sqlite")):
        answer = input(
            "Please provide project admin email address.  This person will be the super user of the dashboard and this project.\n"
        )
        print("generating random password")
        pwd = generate_password(8)
        print(f"Project admin credential is {answer} and the password is {pwd}")
        environment.update({"FL_CREDENTIAL": f"{answer}:{pwd}"})
    client = docker.from_env()
    try:
        container_obj = client.containers.run(
            dashboard_image,
            entrypoint=["/usr/local/bin/python3", "flare/dashboard/wsgi.py"],
            detach=True,
            auto_remove=True,
            name="flare-dashboard",
            ports={8443: port},
            volumes={folder: {"bind": "/var/tmp/flare/dashboard", "model": "rw"}},
            environment=environment,
        )
    except docker.errors.APIError as e:
        print(f"Either {dashboard_image} image does not exist or another flare-dashboard instance is still running.")
        print("Please either provide an existing container image or stop the running container instance.")
        print(e)
        exit(1)
    if container_obj:
        print("Dashboard container started")
        print("Container name flare-dashboard")
        print(f"id is {container_obj.id}")
    else:
        print("Container failed to start")


def stop():
    client = docker.from_env()
    try:
        container_obj = client.containers.get("flare-dashboard")
    except docker.errors.NotFound:
        print("No flare-dashboard container found")
        exit(0)
    container_obj.kill(signal=signal.SIGINT)
    print("flare-dashboard exited")


def main():
    parser = argparse.ArgumentParser()
    define_dashboard_parser(parser)
    args = parser.parse_args()
    handle_dashboard(args)


def define_dashboard_parser(parser):
    parser.add_argument("--start", action="store_true", help="start dashboard")
    parser.add_argument("--stop", action="store_true", help="stop dashboard")
    parser.add_argument("-p", "--port", type=str, default="443", help="port to listen")
    parser.add_argument(
        "-f", "--folder", type=str, help="folder containing necessary info (default: current working directory)"
    )
    parser.add_argument(
        "-i", "--dashboard_image", default="flare/flare", help="container image for running dashboard"
    )
    parser.add_argument(
        "--passphrase", help="Passphrase to encrypt/decrypt root CA private key.  !!! Do not share it with others. !!!"
    )
    parser.add_argument("-e", "--env", action="append", help="additonal environment variables: var1=value1")


def handle_dashboard(args):
    port = args.port
    folder = args.folder
    dashboard_image = args.dashboard_image
    if args.stop:
        stop()
    elif args.start:
        start(port, folder, dashboard_image, args.env, args.passphrase)


if __name__ == "__main__":
    main()
