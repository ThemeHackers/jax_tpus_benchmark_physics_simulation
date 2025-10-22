#!/bin/bash

pip install --upgrade pip
pip install torch==2.1.0 torch_xla==2.1.0 -f https://storage.googleapis.com/pytorch-tpu-releases/wheels/tpuvm/torch_xla-2.1.html
pip install "transformers<5.8"
pip install jax>=0.6.0 flax orbax-checkpoint clu tensorflow-datasets tensorflow-metadata protobuf<4
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install tensorflow tensorflow-metadata psutil rich
pip check

