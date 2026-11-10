import os
import yaml
from pathlib import Path


def load_config(path: str = "config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Allow the LLM API key to come from the environment instead of being
    # stored in plaintext in config.yaml. The env var takes precedence when
    # set, so you can keep secrets out of the config file entirely.
    env_key = os.environ.get("MINIMAX_API_KEY") or os.environ.get("LLM_API_KEY")
    if env_key:
        config.setdefault("llm", {})["api_key"] = env_key

    return config
