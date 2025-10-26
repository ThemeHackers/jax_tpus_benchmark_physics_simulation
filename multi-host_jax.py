import jax
import jax.numpy as jnp
import numpy as np
import os

from jax import pjit
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.distributed import initialize as jax_distributed_initialize

try:
    jax_distributed_initialize()
    print("JAX Distributed Initialized.")
except Exception as e:
    print(f"JAX Distributed FAILED: {e}. Running as single host.")

process_count = jax.process_count()
process_index = jax.process_index()
local_device_count = jax.local_device_count()
global_device_count = jax.device_count()

print(f"Host ID: {process_index} / Total Hosts: {process_count}")
print(f"Local Devices: {local_device_count}, Global Devices: {global_device_count}")

mesh_shape = (process_count, local_device_count)
devices = np.array(jax.devices()).reshape(mesh_shape)

mesh = Mesh(devices, axis_names=('data', 'model'))
print(f"Device Mesh created with shape {mesh_shape} and axes ('data', 'model')")

def train_step(params, batch_data, batch_labels):
    logits = jnp.dot(batch_data, params['w']) + params['b']
    
    batch_size = batch_labels.shape[0]
    labels_one_hot = jax.nn.one_hot(batch_labels, num_classes=10)
    loss = -jnp.mean(jnp.sum(labels_one_hot * jax.nn.log_softmax(logits), axis=-1))

    loss = jax.lax.pmean(loss, axis_name='data')
    return loss

params_spec = {
    'w': P(None, 'model'), 
    'b': P('model')
}

data_spec = P('data', None)
labels_spec = P('data')

p_train_step = pjit(
    train_step,
    in_shardings=(params_spec, data_spec, labels_spec),
    out_shardings=None
)

GLOBAL_BATCH_SIZE = 1024 * process_count
NUM_FEATURES = 100
NUM_CLASSES = 10

params = {
    'w': jnp.ones((NUM_FEATURES, NUM_CLASSES)),
    'b': jnp.ones((NUM_CLASSES,))
}

global_data = None
global_labels = None
if process_index == 0:
    print(f"Host 0 creating Global Data (Batch={GLOBAL_BATCH_SIZE})...")
    global_data = np.random.rand(GLOBAL_BATCH_SIZE, NUM_FEATURES).astype(np.float32)
    global_labels = np.random.randint(0, NUM_CLASSES, (GLOBAL_BATCH_SIZE,)).astype(np.int32)
    print("Host 0 finished creating data.")

print(f"Host {process_index} entering Mesh context to shard data...")

with mesh:
    params_sharding = jax.sharding.NamedSharding(mesh, params_spec)
    data_sharding = jax.sharding.NamedSharding(mesh, data_spec)
    labels_sharding = jax.sharding.NamedSharding(mesh, labels_spec)
    
    sharded_params = jax.device_put(params, params_sharding)
    sharded_data = jax.device_put(global_data, data_sharding)
    sharded_labels = jax.device_put(global_labels, labels_sharding)
    
    jax.block_until_ready(sharded_data)
    print(f"Host {process_index} finished sharding. Starting training step...")

    loss = p_train_step(sharded_params, sharded_data, sharded_labels)

    jax.block_until_ready(loss)

    print(f"Host {process_index} finished. Loss (averaged across all hosts): {loss}")
