from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from . import geometry_util, mode_util


@runtime_checkable
class SimulationBackend(Protocol):
    """Backend-neutral operations used by component setup and result functions."""

    def switch_to_layout(self) -> None: ...

    def clear_geometry(self) -> None: ...

    def clear_layer_geometries(self) -> None: ...

    def load_layer_stack(
        self,
        filename: str | Path,
        size: Sequence[float | None],
        center: Sequence[float] = (0.0, 0.0, 0.0),
        gds_center: Sequence[float] | None = None,
    ) -> None: ...

    def set_layer_geometries(self, layer: str, geometries: Sequence[np.ndarray]) -> None: ...

    def configure_layer(self, layer: str, **params: Any) -> None: ...

    def get_layer_info(self, layer: str) -> dict[str, Any]: ...

    def configure_eme(self, **params: Any) -> None: ...

    def configure_mesh(self, **params: Any) -> None: ...

    def configure_eme_profile(self, **params: Any) -> None: ...

    def run_eme(self, filename: str | None = None) -> None: ...

    def eme_propagation(self, update: bool = False) -> np.ndarray: ...

    def expand_eme_cell_group(self, cell_group_number: int) -> slice: ...

    def set_eme_group_lengths(self, cell_range: slice | int, lengths: Any) -> Any: ...


class LumericalModeBackend:
    """MODE backend that maps domain operations to the ``lumapi`` API."""

    def __init__(self, sim: Any):
        self.sim = sim

    def switch_to_layout(self) -> None:
        self.sim.switchtolayout()

    def clear_geometry(self) -> None:
        self.sim.switchtolayout()
        self.sim.deleteall()

    def clear_layer_geometries(self) -> None:
        self.sim.setnamed("layer group", "geometry", {})

    def load_layer_stack(
        self,
        filename: str | Path,
        size: Sequence[float | None],
        center: Sequence[float] = (0.0, 0.0, 0.0),
        gds_center: Sequence[float] | None = None,
    ) -> None:
        geometry_util.add_layerstack_fromfile(
            self.sim,
            filename=filename,
            size=list(size),
            center=list(center),
            gds_center=list(gds_center) if gds_center is not None else None,
        )

    def set_layer_geometries(self, layer: str, geometries: Sequence[np.ndarray]) -> None:
        geometry_util.set_geometries_to_layer(self.sim, geometries, layer)

    def configure_layer(self, layer: str, **params: Any) -> None:
        self.sim.select("layer group")
        for name, value in params.items():
            self.sim.setlayer(layer, name.replace("_", " "), value)

    def get_layer_info(self, layer: str) -> dict[str, Any]:
        return geometry_util.get_layer_info(self.sim, layer)

    def configure_eme(self, **params: Any) -> None:
        mode_util.define_eme(self.sim, **params)

    def configure_mesh(self, **params: Any) -> None:
        geometry_util.define_mesh(self.sim, **params)

    def configure_eme_profile(self, **params: Any) -> None:
        mode_util.define_eme_profile(self.sim, **params)

    def run_eme(self, filename: str | None = None) -> None:
        mode_util.run_eme(self.sim, filename)

    def eme_propagation(self, update: bool = False) -> np.ndarray:
        return mode_util.run_eme_propagation(self.sim, update=update)

    def expand_eme_cell_group(self, cell_group_number: int) -> slice:
        return mode_util.expand_cell_group(self.sim, cell_group_number)

    def set_eme_group_lengths(self, cell_range: slice | int, lengths: Any) -> Any:
        return mode_util.set_eme_group_lengths(self.sim, cell_range, lengths)