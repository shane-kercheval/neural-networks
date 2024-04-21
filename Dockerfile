FROM python:3.11

# Install necessary commands for adding a PPA
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    zsh \
    lsof

# Set default shell to zsh (optional, depends on your preference)
RUN echo "dash dash/sh boolean false" | debconf-set-selections && \
    dpkg-reconfigure dash

# Install GCC, G++ and make
RUN apt-get install -y --no-install-recommends \
    gcc g++ make

# Clean up to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /code
ENV PYTHONPATH "${PYTHONPATH}:/code"

ENV PYDEVD_WARN_EVALUATION_TIMEOUT 1000

RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

RUN pip3 install torch torchvision torchaudio
