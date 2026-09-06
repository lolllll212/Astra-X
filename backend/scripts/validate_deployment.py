#!/usr/bin/env python3
"""Pre-deployment validation script.

Checks that the environment is safely configured for production before
deploying. Exits with code 0 on success or 1 on failure.

Usage:
    ASTRA_ENVIRONMENT=production python scripts/validate_deployment.py
"""

from __future__ import annotations

import os
import re
import sys


def _check(ok: bool, message: str) -> bool:
    """Print a check result and return whether it passed."""
    symbol = "PASS" if ok else "FAIL"
    print(f"  [{symbol}] {message}")
    return ok


def main() -> int:
    prefix = "ASTRA_"
    env = {k: v for k, v in os.environ.items() if k.startswith(prefix)}

    print("Astra X — Pre-deployment validation")
    print(f"Environment: {env.get('ASTRA_ENVIRONMENT', '(not set)')}")
    print()

    checks: list[bool] = []
    all_passed = True

    # 1. Environment is production.
    env_val = env.get("ASTRA_ENVIRONMENT", "")
    ok = env_val == "production"
    checks.append(_check(ok, "ASTRA_ENVIRONMENT is 'production'"))
    if not ok:
        print("       Set ASTRA_ENVIRONMENT=production in the deployment environment.")

    # 2. Debug is off.
    debug = env.get("ASTRA_DEBUG", "true")
    ok = debug.lower() not in ("true", "1", "yes")
    checks.append(_check(ok, "ASTRA_DEBUG is not enabled"))

    # 3. Secret key is a 64-char hex string.
    secret = env.get("ASTRA_SECRET_KEY", "")
    hex_pattern = re.compile(r"^[0-9a-f]{64}$")
    ok = bool(hex_pattern.match(secret))
    checks.append(_check(ok, "ASTRA_SECRET_KEY is a 64-character hex string"))
    if not ok and len(secret) > 0:
        print(f"       Got {len(secret)} chars; expected 64.")

    # 4. Allowed hosts does not contain wildcard.
    hosts = env.get("ASTRA_ALLOWED_HOSTS", "")
    ok = "*" not in hosts
    checks.append(_check(ok, "ASTRA_ALLOWED_HOSTS does not contain '*'"))

    # 5. CORS origins does not contain wildcard.
    cors = env.get("ASTRA_CORS_ORIGINS", "")
    ok = "*" not in cors
    checks.append(_check(ok, "ASTRA_CORS_ORIGINS does not contain '*'"))

    # 6. Database URL uses PostgreSQL (not SQLite).
    db_url = env.get("ASTRA_DATABASE_URL", "")
    ok = db_url.startswith("postgresql+asyncpg://")
    checks.append(_check(ok, "ASTRA_DATABASE_URL uses postgresql+asyncpg driver"))
    if not ok:
        print(f"       Got: {db_url.split('://')[0] if '://' in db_url else '(invalid)'}")

    # 7. Database echo is off.
    echo = env.get("ASTRA_DATABASE_ECHO", "true")
    ok = echo.lower() not in ("true", "1", "yes")
    checks.append(_check(ok, "ASTRA_DATABASE_ECHO is not enabled"))

    print()
    total = len(checks)
    passed = sum(1 for c in checks if c)
    failed = total - passed
    print(f"Result: {passed}/{total} passed, {failed} failed")

    if passed < total:
        print("\nSome checks failed. Review the configuration before deploying.")
        return 1

    print("All checks passed. Ready for deployment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
