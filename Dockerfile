# Environment for building and running the simulations and the evaluation.
# The repository is not copied into the image but mounted at runtime (see README).
FROM ubuntu:24.04
# The setenv scripts of OMNeT++ and INET require bash
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG OMNETPP_VERSION=6.2.0
ARG INET_VERSION=4.5.4

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential clang lld bison flex perl python3 python3-venv python3-pip \
        libxml2-dev zlib1g-dev pkg-config ca-certificates curl make tmux \
    && rm -rf /var/lib/apt/lists/*

# OMNeT++ (command line only, no GUI). Its configure script requires the Python
# packages of OMNeT++, which are installed in a separate virtual environment.
WORKDIR /opt
RUN curl -fsSL https://github.com/omnetpp/omnetpp/releases/download/omnetpp-${OMNETPP_VERSION}/omnetpp-${OMNETPP_VERSION}-linux-x86_64.tgz | tar -xz \
    && cd omnetpp-${OMNETPP_VERSION} \
    && python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir -r python/requirements.txt \
    && . .venv/bin/activate \
    && . ./setenv \
    && ./configure WITH_QTENV=no WITH_OSG=no WITH_OSGEARTH=no WITH_SCAVE_PYTHON_BINDINGS=no \
    && make -j"$(nproc)" MODE=release base
ENV OMNETPP_ROOT=/opt/omnetpp-${OMNETPP_VERSION}
ENV PATH=${OMNETPP_ROOT}/bin:${PATH}

# INET
RUN curl -fsSL https://github.com/inet-framework/inet/releases/download/v${INET_VERSION}/inet-${INET_VERSION}-src.tgz | tar -xz \
    && mv inet4.5 inet-${INET_VERSION} \
    && cd inet-${INET_VERSION} \
    && . ./setenv \
    && make makefiles \
    && make -j"$(nproc)" MODE=release
ENV INET_DIR=/opt/inet-${INET_VERSION}

# Python packages of the evaluation
COPY evaluation/requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt
ENV PATH=/opt/venv/bin:${PATH}
# Used by evaluation/Makefile instead of a virtual environment created on the host
ENV PYTHON=/opt/venv/bin/python3
# Writable cache for matplotlib when running as a non-root user
ENV MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /work
