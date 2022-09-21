FROM python:3.8
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install zip
COPY flare /opt/Flare/flare
COPY setup.py /opt/Flare/setup.py
COPY versioneer.py /opt/Flare/versioneer.py
COPY README.md /opt/Flare/README.md
COPY setup.cfg /opt/Flare/setup.cfg
WORKDIR /opt/Flare
RUN pip install -U pip
RUN pip install -e .

