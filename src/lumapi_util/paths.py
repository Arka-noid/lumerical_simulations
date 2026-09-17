from pathlib import Path

def get_project_root() -> Path:
    
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Project root not found")


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
STACK_DATA_DIR = DATA_DIR / "stack"