"""`kgx-gaps check <dir>` — run the conformance suite against any KGX export."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kgx-gaps", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="conformance-check a directory of nodes/edges/gaps TSVs")
    c.add_argument("directory", type=Path)
    c.add_argument("--input-rows", type=int, default=None,
                   help="row count of the source evidence table, to check conservation (C8)")
    c.add_argument("--merged", type=int, default=0,
                   help="input rows that repeated a row already written and were written once; the "
                        "producer reports this (Exporter.merged) and C8 counts it")
    a = ap.parse_args(argv)

    from . import conformance
    rep = conformance.check_dir(a.directory, mapping=None, n_input_rows=a.input_rows,
                                n_merged=a.merged)
    print(f"\n  kgx-gaps conformance — {a.directory}\n")
    print(rep)
    print("\n  SPEC.md §5 defines these checks. Skips are not passes: C3 needs a Mapping and C8 an"
          "\n  input row count, neither of which can be inferred from the TSVs alone.\n")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
