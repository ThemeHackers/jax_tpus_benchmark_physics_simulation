#!/bin/bash
set -euo pipefail

echo "Starting installation script..."
echo "Updating system and installing necessary build dependencies..."
sudo apt update -y
sudo apt upgrade -y

sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget

sudo apt install -y python3-dev


export PYTHON_VERSION="3.13.9"
export PYTHON_PATH="/opt/python-$PYTHON_VERSION"

echo "Downloading and installing Python $PYTHON_VERSION..."
wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz
tar -xf Python-$PYTHON_VERSION.tgz
cd Python-$PYTHON_VERSION

./configure --enable-optimizations --prefix=$PYTHON_PATH
make -j$(nproc)
sudo make install
cd ..
rm -rf Python-$PYTHON_VERSION Python-$PYTHON_VERSION.tgz

export PATH="$PYTHON_PATH/bin:$PATH"

echo "Python $PYTHON_VERSION installed to $PYTHON_PATH"

export VENV_NAME=".venv"
echo "Creating and activating virtual environment: $VENV_NAME"

"$PYTHON_PATH/bin/python3" -m venv "$VENV_NAME"

source "$VENV_NAME/bin/activate"

echo "Active Python version:"
python --version

echo "Installing project dependencies (PyTorch/XLA, JAX/TPU, etc.)..."

pip install --upgrade pip


pip install torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html

pip install "transformers<5.8"


pip install jax>=0.7.1 flax orbax-checkpoint clu tensorflow-datasets tensorflow-metadata protobuf<4

pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

pip install psutil rich matplotlib pandas

echo "Performing final pip check..."
pip check

echo "Installation complete. The virtual environment '$VENV_NAME' is active."
echo "Note: For TPU runtime, ensure you're on a Google Cloud TPU VM or Colab with TPU enabled."
echo "Latest versions checked as of Oct 2025: Python 3.13.9, PyTorch/XLA 2.8, JAX 0.7.1."


