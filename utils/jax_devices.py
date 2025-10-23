#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
from rich.table import Table
from rich.console import Console 


console = Console() 
devices = jax.devices()

def list_jax_devices():
    table = Table(title="JAX Devices", show_header=True, header_style="bold magenta")
    table.add_column("Index", style="dim", width=6)
    table.add_column("Device Type", style="cyan", width=12)
    table.add_column("Device ID", style="green", width=10)
    table.add_column("Host ID", style="yellow", width=10)
    table.add_column("Platform", style="blue", width=15)

    for idx, device in enumerate(devices):
        table.add_row(
            str(idx),
            str(device.device_kind),
            str(device.id),
            str(device.host_id),
            str(device.platform),
        )

    return table

console.print(list_jax_devices())
