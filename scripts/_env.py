"""
_env.py
-------
Load API keys from a `.env` file at the project root into os.environ.

Every script in this project reads its credentials straight from os.environ
(ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, GEMINI_API_KEY, GOOGLE_API_KEY,
OPENROUTER_API_KEY), which means a key has to be exported before the script
runs. That is fine on the pod, where setup_pod.sh writes /workspace/env.sh, but
awkward on Windows where it means `setx` plus a new terminal.

This module closes that gap with no new dependency: import it and the .env file
is read. There is no python-dotenv requirement - it is a dozen lines of parsing.

Two deliberate behaviours:

  * A REAL environment variable always wins. The file only fills in what is
    missing, so `setx` / `export` / a CI secret is never silently overridden by a
    stale file, and the pod keeps behaving exactly as it does today.
  * The format is `source`-able. Lines may be written `KEY=value` or
    `export KEY=value`, so the SAME file works with `. .env` in Git Bash and can
    be appended to /workspace/env.sh unchanged.

Usage - one line, as early as possible:

    import _env  # noqa: F401  (loads .env into os.environ)

Never commit the .env file. .gitignore excludes it; .env.example is the shareable
template.
"""
import os

# This file lives in <project>/scripts/, so the project root is one level up.
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE, '.env')


def load(path=ENV_PATH, override=False):
    """Read `path` into os.environ and return the names of the keys it set.

    Blank lines and `#` comments are skipped, as are entries with an empty value,
    so a template file with `OPENROUTER_API_KEY=` waiting to be filled in is
    harmless rather than a key set to the empty string.
    """
    if not os.path.exists(path):
        return []

    loaded = []
    # utf-8-sig so a BOM from a Windows editor does not end up inside the first key.
    with open(path, encoding='utf-8-sig') as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[len('export '):].lstrip()
            if '=' not in line:
                continue
            name, _, value = line.partition('=')
            name = name.strip()
            value = value.strip()
            # Strip one matching pair of surrounding quotes, if present.
            if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
                value = value[1:-1]
            if not name or not value:
                continue
            if name in os.environ and not override:
                continue          # a real environment variable always wins
            os.environ[name] = value
            loaded.append(name)
    return loaded


# Loading on import is the whole point: `import _env` should be enough.
LOADED = load()


if __name__ == '__main__':
    # `python scripts/_env.py` reports what the file provides, without ever
    # printing a secret - only its name and length.
    print(f'.env path : {ENV_PATH}')
    print(f'exists    : {os.path.exists(ENV_PATH)}')
    if LOADED:
        print('loaded from .env:')
        for name in LOADED:
            print(f'  {name:24s} len {len(os.environ[name])}')
    else:
        print('loaded from .env: (nothing - file missing, empty, or every key '
              'already set in the environment)')
    print('\nkey-like variables now visible:')
    for name in sorted(os.environ):
        if 'KEY' in name.upper() or 'TOKEN' in name.upper():
            print(f'  {name:24s} len {len(os.environ[name])}')
