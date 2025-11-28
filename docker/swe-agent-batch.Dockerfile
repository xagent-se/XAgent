FROM python:3.11.10-bullseye

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /

# Install swe-rex for faster startup
RUN pip install pipx
RUN pipx install swe-rex
RUN pipx ensurepath

# Ensure pipx binaries are in PATH
ENV PATH="$PATH:/root/.local/bin/"

# Install any additional tools that might be needed
RUN pip install flake8 pytest

SHELL ["/bin/bash", "-c"]
