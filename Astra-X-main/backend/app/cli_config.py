"""Config management for the ``astra`` CLI.

Provides ``astra config`` subcommands to view, set, and reset
configuration values (backed by the pydantic-settings Settings object).
"""

from __future__ import annotations

import os
from typing import Any

from app.config.settings import get_settings

__all__ = ["cmd_config_get", "cmd_config_list", "cmd_config_reset", "cmd_config_set"]


def _settings_to_dict() -> dict[str, Any]:
    """Return a flattened dict of all settings (excluding secrets)."""
    settings = get_settings()
    result: dict[str, Any] = {}
    for field_name, field_info in settings.model_fields.items():
        value = getattr(settings, field_name)
        if (isinstance(value, str) and field_info.annotation is str) or isinstance(value, (int, float, bool)) or isinstance(value, list):
            result[field_name] = value
        elif isinstance(value, str) and field_name in ("secret_key",):
            result[field_name] = "***"
        else:
            result[field_name] = str(value)
    return result


def cmd_config_get(key: str) -> int:
    """Get a single config value by key."""
    settings = get_settings()
    if not hasattr(settings, key):
        print(f"Unknown config key: {key}")
        return 1
    value = getattr(settings, key)
    if key == "secret_key":
        print(f"{key} = ***")
    else:
        print(f"{key} = {value}")
    return 0


def cmd_config_list() -> int:
    """List all configuration values."""
    settings = get_settings()
    config = _settings_to_dict()
    print()
    header = f"{'Key':<30} {'Value'}"
    print(header)
    print("-" * len(header))
    for k, v in config.items():
        val_str = str(v)
        if len(val_str) > 80:
            val_str = val_str[:77] + "..."
        print(f"  {k:<28} {val_str}")
    print()
    return 0


def cmd_config_set(key: str, value: str) -> int:
    """Set a config value in the .env file."""
    env_path = ".env"
    import os
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            pass

    key_upper = f"ASTRA_{key.upper()}"
    with open(env_path) as f:
        lines = f.readlines()

    found = False
    new_lines = []
    for line in lines:
        if line.startswith(f"{key_upper}="):
            new_lines.append(f"{key_upper}={value}\n")
            found = True
        else:
            new_lines.append(line)

    if not found:
        new_lines.append(f"{key_upper}={value}\n")

    with open(env_path, "w") as f:
        f.writelines(new_lines)

    print(f"Set {key} = {value} in {env_path}")
    return 0


def cmd_config_reset() -> int:
    """Reset .env to defaults."""
    env_path = ".env"
    if not os.path.exists(env_path):
        print("No .env file found.")
        return 0
    os.remove(env_path)
    print("Reset .env to defaults. Restart the server to apply.")
    return 0
