from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"

_PATH_KEYS = [("chroma", "persist_dir"), ("data", "input_dir"), ("eval", "gold_qa_path")]


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for section, key in _PATH_KEYS:
        cfg[section][key] = (path.parent / cfg[section][key]).resolve()
    return cfg
