# Copyright (c) 2026 fy2ne (https://xasp.fy2ne.me). All rights reserved.
# Dual-Licensed:
#   1. Noncommercial Evaluation & Academic Research: PolyForm-Noncommercial-1.0.0
#   2. Commercial / Production Deployment: Requires a paid Commercial License from fy2ne.
#      Source-available on https://github.com/fy2ne/xasp (not distributed via crates.io).

from dataclasses import dataclass


@dataclass
class StagingSlot:
    slot_idx: int
    layer_id: int
    released: bool = False


class StagingLayerHook:
    """Adapter tracking deterministic tpu-sync StagingLayerManager slot release at tau_l."""

    def __init__(self, num_staging_layers: int = 4):
        self.num_staging_layers = num_staging_layers
        self.active: dict[int, StagingSlot] = {}
        self.released_count = 0

    def acquire(self, layer_id: int) -> StagingSlot:
        slot_idx = layer_id % self.num_staging_layers
        slot = StagingSlot(slot_idx=slot_idx, layer_id=layer_id)
        self.active[layer_id] = slot
        return slot

    def release_at_deadline(self, layer_id: int) -> None:
        slot = self.active.pop(layer_id, None)
        if slot is not None:
            slot.released = True
            self.released_count += 1
