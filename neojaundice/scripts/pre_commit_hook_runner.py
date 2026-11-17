import importlib
import sys

if len(sys.argv) < 2:
    raise SystemExit("Usage: pre_commit_hook_runner.py MODULE [ARGS...]")

module_name = sys.argv[1]
hook_args = sys.argv[2:]

try:
    module = importlib.import_module(module_name)
except ImportError as exc:
    print(f"Could not import hook {module_name}: {exc}", file=sys.stderr)
    raise SystemExit(1)

raise SystemExit(module.main(hook_args))
