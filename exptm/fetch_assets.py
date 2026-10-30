"""Fetch the three pinned inputs needed for the archived checkpoint replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent


def matches(path, asset):
    if not path.is_file() or path.stat().st_size != asset["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == asset["sha256"]


def ensure_assets(check_only=False):
    assets = json.loads((ROOT / "assets.json").read_text())
    for asset in assets:
        target = ROOT / "author_source" / asset["path"]
        if target.exists():
            if not matches(target, asset):
                raise RuntimeError(f"Hash/size mismatch: {target}; existing file left unchanged.")
            print("Verified:", asset["path"], flush=True)
            continue
        if check_only:
            raise FileNotFoundError(f"Missing {target}. Run: python fetch_assets.py")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            print("Downloading:", asset["path"], flush=True)
            request = Request(asset["source_url"], headers={"User-Agent": "ExpTM-replay"})
            with urlopen(request, timeout=120) as response, tempfile.NamedTemporaryFile(
                dir=target.parent, suffix=".download", delete=False
            ) as stream:
                temporary = Path(stream.name)
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if not matches(temporary, asset):
                raise RuntimeError(f"Downloaded asset failed verification: {asset['path']}")
            # Same-directory hard link publishes the verified file atomically,
            # and fails rather than replacing a concurrent writer's target.
            os.link(temporary, target)
            print("Verified:", asset["path"], flush=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    ensure_assets(parser.parse_args().check_only)
