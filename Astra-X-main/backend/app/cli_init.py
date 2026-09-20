"""Project initialization for Astra X.

Provides ``astra init`` to scaffold a new Astra X project with
configuration files, workspace directories, and default tools.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["cmd_init"]


PROJECT_STRUCTURE = {
    ".astra/config.json": r'''{
  "model": "qwen2.5-coder-7b-instruct",
  "provider": "lm_studio",
  "workspace": "./workspace",
  "native_tools": false,
  "tool_schema_role": "user",
  "max_iterations": 5,
  "log_level": "INFO"
}
''',
    ".env.example": r'''# Astra X Environment
ASTRA_ENVIRONMENT=development
ASTRA_DEBUG=true
ASTRA_DEFAULT_LLM_MODEL=qwen2.5-coder-7b-instruct
ASTRA_LM_STUDIO_BASE_URL=http://localhost:1234/v1
ASTRA_DATABASE_URL=sqlite+aiosqlite:///./astra_x.db
ASTRA_SECRET_KEY=change-me-to-a-64-char-hex-string
ASTRA_ALLOWED_HOSTS=*
ASTRA_CORS_ORIGINS=http://localhost:5173
ASTRA_LOG_LEVEL=INFO
ASTRA_MAX_ITERATIONS=5
''',
    "workspace/.gitkeep": "",
    "README.md": r'''# Astra X Project

Astra X project. Edit `.astra/config.json` to configure the model,
tools, and workspace settings.

## Quick Start

```bash
astra serve          # Start the backend server
astra chat "Hello"   # One-shot chat
astra repl           # Interactive session
astra init           # Re-initialize the project
```

## Config

Edit `.astra/config.json` or set environment variables in `.env`.
''',
    ".gitignore": r'''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
.venv/
venv/

# Database
*.db
*.sqlite

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
''',
}


def cmd_init(path: str = ".") -> int:
    """Initialize a new Astra X project."""
    root = Path(path).resolve()

    if (root / ".astra").exists():
        print(f"  Astra X project already exists at {root}.")
        print("  Use `astra init --force` to overwrite config files.")
        return 1

    print(f"  Initializing Astra X project at {root}...")

    # Create directories
    for file_path in PROJECT_STRUCTURE:
        full = root / file_path
        full.parent.mkdir(parents=True, exist_ok=True)

    # Write files
    for file_path, content in PROJECT_STRUCTURE.items():
        full = root / file_path
        full.write_text(content.strip() + "\n", encoding="utf-8")
        print(f"  Created {file_path}")

    # Create .astra directory explicitly
    (root / ".astra").mkdir(parents=True, exist_ok=True)

    print()
    print("  Astra X project initialized!")
    print("  Next steps:")
    print(f"    cd {root}")
    print("    cp .env.example .env")
    print("    astra serve")
    print()
    return 0
