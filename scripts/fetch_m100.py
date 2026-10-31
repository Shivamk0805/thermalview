"""Download only the M100 metrics ThermalView needs, straight out of a Zenodo .tar.

A monthly M100 archive is one big uncompressed .tar (several GB). ThermalView
reads about 60 of its ~340 metric files. This script reads the tar's headers
with HTTP Range requests (a few KB each), picks the metric files that match
the patterns in config.toml [m100], and downloads just those, each split into
chunks fetched over several connections in parallel. It resumes where it
stopped if interrupted. Standard library only.

  python scripts\\fetch_m100.py 21-02                 # month, as in the Zenodo file name
  python scripts\\fetch_m100.py 21-02 --list          # only show what would be downloaded
  python scripts\\fetch_m100.py 21-02 --workers 12    # more parallel connections

Files land in data/raw/m100/<path inside the tar>, exactly where `tar -xf`
would have put them, so `python -m thermalview probe` and `all` work unchanged.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORDS = {  # M100 ExaData on Zenodo (CC BY 4.0)
    **{f"20-{m:02d}": "7588815" for m in range(3, 13)},
    **{f"21-{m:02d}": "7589131" for m in range(1, 7)},
}
CHUNK = 8 * 1024 * 1024
BLOCK = 64 * 1024          # bytes read per header request (neighbouring headers come free)
UA = {"User-Agent": "thermalview-fetch/1.0"}


def get_range(url: str, start: int, end: int, tries: int = 8) -> bytes:
    """Bytes [start, end] inclusive, retrying with backoff. Requires a server that honours Range."""
    for attempt in range(tries):
        req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 206:
                    raise RuntimeError(f"server ignored the Range header (HTTP {r.status}); "
                                       "use a download manager on the whole file instead")
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError(f"short read {len(data)} of {end - start + 1}")
            return data
        except RuntimeError:
            raise
        except Exception:  # network hiccup: wait and retry
            if attempt == tries - 1:
                raise
            time.sleep(min(60, 2 ** attempt))
    raise IOError("unreachable")


def total_size(url: str) -> int:
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range", "")
        if r.status != 206 or "/" not in cr:
            raise RuntimeError("server does not support byte ranges; download the whole file instead")
        return int(cr.rsplit("/", 1)[1])


def _octal(field: bytes) -> int:
    if field and field[0] & 0x80:                         # GNU base-256 for big sizes
        return int.from_bytes(field[1:], "big")
    s = field.split(b"\0", 1)[0].strip()
    return int(s, 8) if s else 0


def _pax(data: bytes) -> dict[str, str]:
    out, i = {}, 0
    while i < len(data):
        sp = data.index(b" ", i)
        n = int(data[i:sp])
        k, v = data[sp + 1:i + n - 1].decode("utf-8", "replace").split("=", 1)
        out[k] = v
        i += n
    return out


def list_members(url: str, size: int, log=print) -> list[tuple[str, int, int]]:
    """(path, data_offset, data_size) for every regular file, read from headers only."""
    members, off = [], 0
    buf, buf_off = b"", 0
    long_name, pax = None, {}
    t0, n_req = time.time(), 0

    def read(o: int, n: int) -> bytes:
        nonlocal buf, buf_off, n_req
        if not (buf_off <= o and o + n <= buf_off + len(buf)):
            end = min(size - 1, o + max(n, BLOCK) - 1)
            buf, buf_off = get_range(url, o, end), o
            n_req += 1
        return buf[o - buf_off:o - buf_off + n]

    while off + 512 <= size:
        h = read(off, 512)
        if h == b"\0" * 512:
            break
        name = h[0:100].split(b"\0", 1)[0].decode("utf-8", "replace")
        prefix = h[345:500].split(b"\0", 1)[0].decode("utf-8", "replace")
        if prefix:
            name = prefix + "/" + name
        fsize = _octal(h[124:136])
        typ = h[156:157]
        data_off = off + 512
        if typ == b"L":                                  # GNU long name for the next entry
            long_name = read(data_off, fsize).split(b"\0", 1)[0].decode("utf-8", "replace")
        elif typ == b"x":                                # pax header for the next entry
            pax = _pax(read(data_off, fsize))
        elif typ in (b"0", b"\0", b"7"):
            path = pax.get("path") or long_name or name
            fsize = int(pax.get("size", fsize))
            members.append((path, data_off, fsize))
            long_name, pax = None, {}
        else:
            long_name, pax = None, {}
        off = data_off + ((fsize + 511) // 512) * 512
        if n_req and n_req % 100 == 0 and off >= buf_off + len(buf):
            log(f"  index: {len(members)} files, {off / size:.0%} of archive scanned, {time.time() - t0:.0f}s")
    log(f"  index done: {len(members)} files found with {n_req} small requests in {time.time() - t0:.0f}s")
    return members


def _kv(part: str) -> str:
    return part.split("=", 1)[1] if "=" in part else part


def wanted(members, cfg_m100: dict) -> list[tuple[str, int, int]]:
    specs = []
    for field in ("cpu_temp", "load", "power", "inlet", "fan"):
        if field in cfg_m100:
            plugin = cfg_m100.get("load_plugin", cfg_m100["plugin"]) if field == "load" else cfg_m100["plugin"]
            specs.append((plugin, re.compile(cfg_m100[field]["pattern"])))
    out = []
    for path, off, size in members:
        parts = [p for p in path.split("/") if p]
        if len(parts) < 3 or not parts[-1].endswith(".parquet"):
            continue
        plugin, metric = _kv(parts[-3]), _kv(parts[-2])
        if any(plugin == pl and rx.search(metric) for pl, rx in specs):
            out.append((path, off, size))
    return out


def download(url: str, files, dest: Path, workers: int, log=print) -> None:
    jobs, done_bytes = [], 0
    for path, off, size in files:
        target = dest / Path(*[p for p in path.split("/") if p])
        state = target.with_suffix(target.suffix + ".chunks.json")
        if target.exists() and target.stat().st_size == size and not state.exists():
            done_bytes += size
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size != size:
            with open(target, "wb") as fh:
                fh.truncate(size)
        finished = set(json.loads(state.read_text())) if state.exists() else set()
        state.write_text(json.dumps(sorted(finished)))
        for c, s in enumerate(range(0, size, CHUNK)):
            if c in finished:
                done_bytes += min(CHUNK, size - s)
                continue
            jobs.append((target, state, c, off + s, off + min(size, s + CHUNK) - 1, s))
    grand = done_bytes + sum(j[4] - j[3] + 1 for j in jobs)
    log(f"  {len(files)} files, {grand / 1e9:.2f} GB in total, {done_bytes / 1e9:.2f} GB already here; "
        f"{len(jobs)} chunks to fetch with {workers} connections")
    lock = threading.Lock()
    progress = {"bytes": done_bytes, "t0": time.time(), "start": done_bytes}

    def work(job):
        target, state, c, a, b, local = job
        data = get_range(url, a, b)
        with lock:
            with open(target, "r+b") as fh:
                fh.seek(local)
                fh.write(data)
            fin = set(json.loads(state.read_text()))
            fin.add(c)
            state.write_text(json.dumps(sorted(fin)))
            progress["bytes"] += len(data)
        return target, state

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 5 == 0 or i == len(futs):
                el = time.time() - progress["t0"]
                rate = (progress["bytes"] - progress["start"]) / max(el, 1e-9)
                left = (grand - progress["bytes"]) / max(rate, 1)
                log(f"  {progress['bytes'] / 1e9:6.2f} / {grand / 1e9:.2f} GB  {rate / 1e6:5.2f} MB/s  "
                    f"about {left / 60:.0f} min left")
    for path, _off, size in files:                    # chunk files mark incomplete downloads
        target = dest / Path(*[p for p in path.split("/") if p])
        state = target.with_suffix(target.suffix + ".chunks.json")
        n_chunks = (size + CHUNK - 1) // CHUNK
        if state.exists() and len(json.loads(state.read_text())) == n_chunks:
            state.unlink()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("month", help="e.g. 21-02")
    ap.add_argument("--url", help="archive URL (default: the Zenodo file for that month)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--list", action="store_true", help="only list the files that would be downloaded")
    ap.add_argument("--list-all", action="store_true", help="list every file in the archive (plugins, metrics, sizes)")
    ap.add_argument("--plugin", action="append", default=[],
                    help="also download every metric of this plugin (e.g. a job table); repeatable")
    ap.add_argument("--config", default=str(ROOT / "config.toml"))
    args = ap.parse_args(argv)
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    if args.url:
        url = args.url
    elif args.month in RECORDS:
        url = f"https://zenodo.org/records/{RECORDS[args.month]}/files/{args.month}.tar?download=1"
    else:
        print(f"unknown month {args.month}; pass --url")
        return 2
    dest = ROOT / cfg["paths"]["raw_m100"]
    print(f"archive: {url}")
    try:
        size = total_size(url)
    except Exception as exc:
        print(f"cannot read the archive: {exc}")
        return 1
    print(f"  {size / 1e9:.2f} GB on the server; reading its index …")
    members = list_members(url, size)
    if args.list_all:
        by_plugin = {}
        for p, _o, s in members:
            parts = [x for x in p.split("/") if x]
            key = _kv(parts[-3]) if len(parts) >= 3 else (parts[0] if parts else "?")
            by_plugin.setdefault(key, []).append((p, s))
        for plugin, items in sorted(by_plugin.items()):
            print(f"  [{plugin}] {len(items)} files, {sum(s for _p, s in items) / 1e9:.2f} GB")
            for p, s in items[:6]:
                print(f"      {s / 1e6:8.1f} MB  {p}")
            if len(items) > 6:
                print(f"      … {len(items) - 6} more")
        return 0
    files = wanted(members, cfg["m100"])
    if args.plugin:
        extra = [(p, o, s) for p, o, s in members
                 if any(f"plugin={pl}/" in p or f"/{pl}/" in p for pl in args.plugin) and (p, o, s) not in files]
        files += extra
        print(f"  + {len(extra)} files from plugin(s) {', '.join(args.plugin)}")
    need = sum(s for _p, _o, s in files)
    print(f"  need {len(files)} of {len(members)} files: {need / 1e9:.2f} GB ({need / size:.0%} of the archive)")
    if args.list:
        for p, _o, s in files:
            print(f"    {s / 1e6:8.1f} MB  {p}")
        return 0
    download(url, files, dest, args.workers)
    print(f"done. Files are under {dest}. Next: python -m thermalview probe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
