ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.05-py3
FROM ${FROM_IMAGE_NAME}

# Install Python dependencies
RUN pip install --no-cache-dir https://github.com/mlperf/logging/archive/0.7.0-rc2.zip

# Install DLRM
WORKDIR /workspace/dlrm
COPY . .
