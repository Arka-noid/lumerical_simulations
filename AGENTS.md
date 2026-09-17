# Repository purpose

This repository is a Python framework for photonic component simulation workflows built
on Lumerical products (MODE, FDE, EME, FDTD) via the `lumapi` Python API. It provides:

- `src/lumapi_util/`: reusable simulation, sweep, Monte Carlo, geometry, and plotting utilities.
- `notebooks/components/`: design notebooks for photonic components (couplers, edge couplers,
  MMIs, transitions, waveguides).
- `scripts/stacks/`: tooling to compile Lumerical foundry process/layer-stack files.
- `data/stack/`: layer-stack and process definition files used by the simulations.

## Simulation workflow architecture

Simulation workflows use composable callables, analogous to Lumerical's setup and result
scripts, while remaining fully Python-native and reusable.

- A **setup function** has the contract `setup(sim, **params)`. It switches to layout mode as
    needed and mutates the simulation geometry, solver, mesh, and monitors. It may return a
    dictionary of derived values for an associated result function.
- A **result function** has the contract `result(sim, **params)`. It runs the required analysis
    and returns one well-defined metric or structured result. Result functions must not rebuild
    geometry.
- `lumapi_util.sweep_param_result` composes the two: it merges fixed `geometry_params` with each
    point in `sweep_params`, calls the setup function, then calls the result function. Use
    `pass_geom_output_to_result_fn=True` only when the result needs derived setup output.
- Keep geometry construction, simulation execution, metric extraction, and plotting separate.
    This permits the same geometry to be swept against multiple metrics and the same metric to be
    applied to multiple preconfigured geometries.

### Configuration and presets

- Each generic setup function owns a `default_*_params` dictionary and accepts `**params`.
    Resolve its configuration at the start with `{**default_*_params, **params}`. Do not add a
    formal argument merely to make a setting configurable.
- Geometry entries may also have their own defaults; merge entry defaults with each supplied
    entry before use. Unknown settings should either be deliberately forwarded to the relevant
    utility or rejected with a clear error, never silently ignored.
- Create named, reusable devices by binding defaults with `functools.partial`, for example:
    `si_transition_setup = partial(build_transition, layers=si_layers, wavelength=1.55)`.
    A partial is a preset, not a second implementation; callers and sweeps can still override its
    bound keyword settings.
- Prefer small immutable-like dictionaries/lists for defaults and preset geometry. Do not mutate
    caller-supplied settings or shared default dictionaries.
- Notebook examples should define settings and a setup preset in separate cells, then pass that
    preset directly to `sweep_param_result`. This makes an individual parameter change or a new
    sweep dimension a data change rather than a function rewrite.
