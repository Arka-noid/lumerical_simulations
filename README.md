# Lumerical simulations

Python utilities, notebooks, and scripts for photonic component simulations with Lumerical.

## Requirements

Install and license a Lumerical product that provides the solver and Python API required by your workflow, such as MODE, FDE, EME, or FDTD. The installation must provide the `lumapi` Python module, and a valid Lumerical license must be available when simulations run.

You also need:

- Git
- UV, the Python package and environment manager
- VS Code with the Microsoft Python and Jupyter extensions for notebook work

This guide supports Windows and Linux.

## Install from scratch

### 1. Clone the repository

```bash
git clone https://github.com/Arka-noid/lumerical_simulations.git
cd lumerical_simulations
```

### 2. Install UV

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify the installation:

```bash
uv --version
```

### 3. Locate the Lumerical Python interpreter

Use the Python executable installed with Lumerical so that `lumapi` is available. Use the path shown by your Lumerical installation or documentation, represented below by `<lumerical-python>`.

### 4. Create the UV environment

```bash
uv venv .venv --python <lumerical-python> --system-site-packages
```

The `--system-site-packages` option lets the environment see packages supplied by the Lumerical interpreter, including `lumapi`. If `lumapi` is already available in a normal Python installation, use that interpreter and omit this option.

### 5. Install dependencies with UV

Install the core simulation and analysis dependencies:

```bash
uv sync
```

For notebooks, install the kernel dependency too:

```bash
uv sync --extra notebook
```

For stack-generation workflows using GDS tooling:

```bash
uv sync --extra stack
```

Both optional groups can be installed together:

```bash
uv sync --extra notebook --extra stack
```

`lumapi` is intentionally not a PyPI dependency. It must come from the licensed Lumerical installation.

### 6. Verify the environment

```bash
uv run python -c "import numpy, scipy, lumapi_util; print('Project environment: OK')"
uv run python -c "import lumapi; print('Lumerical API: OK')"
```

If the second command fails, recreate `.venv` with the Python executable supplied by Lumerical. Do not copy licensed Lumerical files into this repository.

## Configure VS Code

1. Open the repository folder in VS Code.
2. Install the Microsoft **Python** and **Jupyter** extensions.
3. Run **Python: Select Interpreter** and choose `.venv`.
4. Open a notebook and select the `.venv` kernel.

The repository includes workspace settings for both the legacy root package and the `src` package layout.

## Run code

Run a Python script through UV:

```bash
uv run python path/to/script.py
```

Run notebooks through VS Code after selecting the `.venv` kernel. Cells that create or run simulations require the Lumerical installation, solver, and license.

## Repository layout

- `lumapi_util/`: legacy simulation utilities
- `src/lumapi_util/`: package layout used by the UV project configuration
- `components/` and `notebooks/`: design notebooks and component data
- `lumerical_scripts/`: Lumerical script files
- `stacks/` and `data/stack/`: process and layer-stack data

## Troubleshooting

### `ModuleNotFoundError: lumapi`

The selected interpreter cannot see the Lumerical API. Recreate `.venv` using the Python executable supplied by Lumerical and `--system-site-packages`, then rerun the verification command.

### VS Code cannot find the notebook kernel

Run `uv sync --extra notebook`, select `.venv` as the Python interpreter, and reload the VS Code window.

### A simulation cannot start

Confirm that the required Lumerical solver is installed, a license is available, and the notebook or script points to existing layer-stack and input files.
