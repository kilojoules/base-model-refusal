#!/usr/bin/env python3
"""Delete orphaned weight blobs from the HuggingFace cache.

Newer huggingface_hub versions enable a shared blob store, marked by
`hub/blobs/.huggingface-shared-blobs`, sharded two characters deep as
`hub/blobs/<ab>/<sha>`. A model's snapshot symlink points at its own `models--X/blobs/`
entry, which is itself a link into that shared store. Removing a `models--*` directory
therefore frees almost nothing: both levels of symlink go, the multi-gigabyte blob stays.
Over a seven-model ladder plus anchors plus judges that is roughly 400GB accumulating
against a 600GB volume.

`os.path.realpath` follows the whole chain, so a snapshot symlink resolves directly to
the shared-store path, and the shared store must be walked recursively rather than listed
at its top level.

A blob is orphaned when no snapshot symlink anywhere in the cache resolves to it.

Two safety rules, because this runs while downloads may be in flight:
  - `.incomplete` files are never touched.
  - a blob must be older than `--min-age-min` before it is eligible. A finished download
    exists briefly before its symlink is created, and deleting inside that window would
    corrupt the model being fetched.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import time


def referenced_blobs(hub: pathlib.Path) -> set[str]:
    refs = set()
    for model_dir in hub.glob("models--*"):
        snap = model_dir / "snapshots"
        if not snap.exists():
            continue
        for p in snap.rglob("*"):
            if p.is_symlink():
                try:
                    refs.add(os.path.realpath(p))
                except OSError:
                    pass
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", default=os.path.expanduser(
        os.environ.get("HF_HOME", "~/.cache/huggingface") + "/hub"))
    ap.add_argument("--min-age-min", type=float, default=20.0)
    ap.add_argument("--min-size-mb", type=float, default=50.0,
                    help="ignore small files; only large weight blobs are worth reaping")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    hub = pathlib.Path(args.hub)
    blobs = hub / "blobs"
    if not blobs.is_dir():
        print(f"[reap] no shared blob store at {blobs}; nothing to do")
        return

    refs = referenced_blobs(hub)
    now = time.time()
    freed = 0
    kept_young = 0
    for b in blobs.rglob("*"):
        if b.is_symlink() or not b.is_file() or b.name.endswith(".incomplete"):
            continue
        if b.name.startswith("."):
            continue
        try:
            stat = b.stat()
        except OSError:
            continue
        if stat.st_size < args.min_size_mb * 1e6:
            continue
        if os.path.realpath(b) in refs:
            continue
        age_min = (now - stat.st_mtime) / 60
        if age_min < args.min_age_min:
            kept_young += 1
            continue
        freed += stat.st_size
        if args.dry_run:
            print(f"[reap] would remove {b.name[:12]} {stat.st_size/1e9:.2f}GB "
                  f"age={age_min:.0f}min")
        else:
            try:
                b.unlink()
            except OSError as e:
                print(f"[reap] failed {b.name[:12]}: {e}")
                freed -= stat.st_size

    import shutil
    free = shutil.disk_usage(str(hub)).free / 1e9
    verb = "would free" if args.dry_run else "freed"
    print(f"[reap] {verb} {freed/1e9:.1f}GB; {kept_young} blobs too young to touch; "
          f"{free:.0f}GB free")


if __name__ == "__main__":
    main()
