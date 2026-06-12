# -*- coding: utf-8 -*-

"""Command-line entrypoint to utils/sphinx

Use this as e.g. `python utils/sphinx --test` to run sphinx smoke tests.
"""

import sys
import argparse
from llvm_sphinx.ext import ghlinks

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="run sphinx self-tests")
    args = parser.parse_args(argv)

    if args.test:
        ghlinks.run_tests()
        print(
            "ghlinks.py: tests passed; next, rebuild docs-clang-html and spot check the release notes"
        )
        return 0

    parser.print_help(sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
