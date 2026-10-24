"""Entrant tool: the public scan around a fiber cube, so a tracer can see past the cube's faces (0.9.0).

Writes the cube's CT padded by MARGIN voxels on every side, from the raw scan the cube was cut from, converted as
villa's fibers-dataset generator made the cube's own CT: uint16 / 257, clipped to 0-255, truncated to uint8. The cube
sits at [MARGIN, MARGIN + size) on each axis. Scoring is unchanged: on the cube alone, against its ground truth.

The raw scans, all public:
  Scroll 1  PHercParis4 20230205180739: one uncompressed 16-bit TIFF per slice, read by HTTP byte range
  Scroll 3  PHerc332 20231117143551: the same
  Scroll 5  PHerc172 20241024131838: a raw uint16 zarr (v2), read chunk by chunk; needs `pip install numcodecs`

Only the needed rows of each slice, or the needed chunks, are fetched, and cached in CACHE; a failed request is tried
again. Each TIFF's header is read
and checked before its rows; any other layout is refused, never guessed. With --check, the padded array's core must
equal the cube's own CT (villa's `.tif`, or a `.npy`) voxel for voxel, or the tool exits 1.

  python tools/context_ct.py CUBE --out PADDED.npy [--margin 128] [--cache DIR] [--check CUBE_CT.tif]

CUBE is the target's stem, `<scroll>_<z>_<y>_<x>_<size>` (e.g. s1_00497_01497_03997_256).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

TIFF_VOLUMES = {
    "s1": {
        "url": "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes/20230205180739",
        "shape": (14376, 7888, 8096),
        "digits": 5,
    },
    "s3": {
        "url": "https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/volumes/20231117143551",
        "shape": (9778, 3550, 3400),
        "digits": 4,
    },
}
ZARR_VOLUMES = {"s5": "https://dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/volumes_zarr/20241024131838.zarr"}
MARGIN = 128  # the context the published scan-context rows used
HEADER_BYTES = 4096
PER_REQUEST = 200  # ranges per request: keeps the Range header near 5 KB


def padded_bounds(cube: str, margin: int = MARGIN):
    """[lo, hi) (z, y, x) of the cube padded by MARGIN, in the scan's voxels."""
    z, y, x, n = (int(v) for v in cube.split("_")[1:5])
    lo = (z - margin, y - margin, x - margin)
    return lo, tuple(v + n + 2 * margin for v in lo)


def to_uint8(raw: np.ndarray, slab: int = 64) -> np.ndarray:
    """villa's fibers-dataset generator, np.clip(volume / 257, 0, 255).astype(np.uint8), a slab at a time."""
    out = np.empty(raw.shape, np.uint8)
    for i in range(0, raw.shape[0], slab):
        out[i : i + slab] = np.clip(raw[i : i + slab].astype(np.float64) / 257, 0, 255).astype(np.uint8)
    return out


# -- raw TIFF slices, by byte range


def parse_header(b: bytes) -> dict:
    """The first IFD of a classic little-endian TIFF: its geometry and single strip, or ValueError."""
    if b[:4] != b"II*\x00":
        raise ValueError("not a little-endian classic TIFF")
    off = struct.unpack("<I", b[4:8])[0]
    n = struct.unpack("<H", b[off : off + 2])[0]
    tags = {}
    for i in range(n):
        tag, typ, cnt, val = struct.unpack("<HHII", b[off + 2 + 12 * i : off + 14 + 12 * i])
        if typ == 3 and cnt == 1:
            val = struct.unpack("<H", b[off + 10 + 12 * i : off + 12 + 12 * i])[0]
        tags[tag] = (typ, cnt, val)
    width, height = tags[256][2], tags[257][2]
    bits, compression = tags[258][2], tags.get(259, (3, 1, 1))[2]
    offsets, counts = tags[273], tags[279]
    if bits != 16 or compression != 1 or tags.get(277, (3, 1, 1))[2] != 1:
        raise ValueError(f"not uncompressed single-channel 16-bit: bits {bits}, compression {compression}")
    if offsets[1] != 1 or counts[1] != 1:
        raise ValueError(f"{offsets[1]} strips: only one strip is read")
    if counts[2] != width * height * 2:
        raise ValueError("the strip is not the whole image")
    return {"width": width, "height": height, "data_offset": offsets[2]}


def parse_byteranges(body: bytes, content_type: str, content_range: str | None) -> dict[tuple[int, int], bytes]:
    """{(start, end): bytes} from a 206 response, single-range or multipart/byteranges."""
    if not content_type.startswith("multipart/byteranges"):
        if not content_range or not content_range.startswith("bytes "):
            raise ValueError("a single-range response without Content-Range")
        a, b = (int(v) for v in content_range.split()[1].split("/")[0].split("-"))
        if len(body) != b - a + 1:
            raise ValueError("body length differs from Content-Range")
        return {(a, b): body}
    boundary = content_type.split("boundary=", 1)[1].strip().strip('"').encode()
    out: dict[tuple[int, int], bytes] = {}
    pos = 0
    marker = b"--" + boundary
    while True:
        pos = body.index(marker, pos) + len(marker)
        if body[pos : pos + 2] == b"--":
            return out
        head_end = body.index(b"\r\n\r\n", pos)
        headers = body[pos:head_end].decode("latin-1").split("\r\n")
        cr = next(h for h in headers if h.lower().startswith("content-range:"))
        a, b = (int(v) for v in cr.split()[2].split("/")[0].split("-"))
        start = head_end + 4
        out[(a, b)] = body[start : start + b - a + 1]
        pos = start + b - a + 1


def retry(fn, *args, tries: int = 6, wait: float = 5.0):
    """A request that failed in any way (a timeout, a reset, a short read) is tried again after 5, 10, 20, ... s.
    The bytes are not trusted for it: --check compares the result with the cube."""
    import time

    for i in range(tries):
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001 (TimeoutError, URLError, http.client.IncompleteRead, ...)
            if i == tries - 1:
                raise
            print(f"retrying after {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            time.sleep(min(120.0, wait * 2**i))


def _get_ranges(url: str, ranges: list[tuple[int, int]]) -> tuple[str, str | None, bytes]:
    return retry(_get_ranges_once, url, ranges)


def _get_ranges_once(url: str, ranges: list[tuple[int, int]]) -> tuple[str, str | None, bytes]:
    spec = ",".join(f"{a}-{b}" for a, b in ranges)
    req = urllib.request.Request(url, headers={"Range": f"bytes={spec}"})
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 (fixed https URLs)
        if r.status != 206:
            raise ValueError(f"{url}: HTTP {r.status}, not a partial response")
        return r.headers.get("Content-Type", ""), r.headers.get("Content-Range"), r.read()


def read_slice_window(url: str, lo, hi, get_ranges=_get_ranges, per_request: int = PER_REQUEST) -> np.ndarray:
    """Rows lo[0]..hi[0]-1 and columns lo[1]..hi[1]-1 of one slice, as uint16."""
    ctype, crange, head = get_ranges(url, [(0, HEADER_BYTES - 1)])
    # A server clips a range past the end of a short file, so take the one part whatever its end.
    info = parse_header(next(iter(parse_byteranges(head, ctype, crange).values())))
    if hi[0] > info["height"] or hi[1] > info["width"] or min(lo) < 0:
        raise ValueError(f"{lo}-{hi} leaves the {info['height']} x {info['width']} slice")
    (y0, x0), (y1, x1) = lo, hi
    w, d = info["width"], info["data_offset"]
    ranges = [(d + 2 * (r * w + x0), d + 2 * (r * w + x1) - 1) for r in range(y0, y1)]
    got: dict[tuple[int, int], bytes] = {}
    for i in range(0, len(ranges), per_request):
        ctype, crange, body = get_ranges(url, ranges[i : i + per_request])
        got.update(parse_byteranges(body, ctype, crange))
    return np.stack([np.frombuffer(got[r], "<u2") for r in ranges])


def read_tiff_region(volume: dict, lo, hi, cache: Path, get_ranges=_get_ranges) -> np.ndarray:
    """[lo, hi) (z, y, x) of a TIFF-slice volume as uint16, each slice's window cached in CACHE."""
    if any(a < 0 or b > s for a, b, s in zip(lo, hi, volume["shape"], strict=True)):
        raise ValueError(f"{lo}-{hi} leaves the volume {volume['shape']}")
    cache.mkdir(parents=True, exist_ok=True)
    out = np.empty(tuple(b - a for a, b in zip(lo, hi, strict=True)), "<u2")
    for i, z in enumerate(range(lo[0], hi[0])):
        f = cache / f"{z}_{lo[1]}_{lo[2]}_{hi[1]}_{hi[2]}.npy"
        if not f.exists():
            win = read_slice_window(f"{volume['url']}/{z:0{volume['digits']}d}.tif", lo[1:], hi[1:], get_ranges)
            tmp = f.with_suffix(".part.npy")
            np.save(tmp, win)
            tmp.rename(f)
        out[i] = np.load(f)
    return out


# -- a raw zarr (v2), chunk by chunk


def _get(url: str) -> bytes:
    return retry(_get_once, url)


def _get_once(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310 (fixed https URLs)
        return r.read()


def read_zarr_region(source: str, lo, hi, cache: Path, get=_get) -> np.ndarray:
    """[lo, hi) (z, y, x) of SOURCE's level 0, from its chunks (cached in CACHE as the bytes served)."""
    meta = json.loads(get(f"{source}/0/.zarray"))
    c = meta["chunks"][0]
    if meta["chunks"] != [c] * 3 or meta.get("dimension_separator") != "/":
        raise ValueError(f"unexpected layout: {meta}")
    if any(a < 0 or b > s for a, b, s in zip(lo, hi, meta["shape"], strict=True)):
        raise ValueError(f"{lo}-{hi} leaves the volume {meta['shape']}")
    if meta["compressor"] is None:
        decode = bytes
    else:
        try:
            from numcodecs import get_codec
        except ImportError as e:
            raise SystemExit("Scroll 5's raw zarr is compressed: pip install numcodecs") from e
        decode = get_codec(meta["compressor"]).decode
    ks = [range(a // c, (b - 1) // c + 1) for a, b in zip(lo, hi, strict=True)]
    base = np.array([k.start for k in ks]) * c
    buf = np.zeros(tuple(len(k) * c for k in ks), np.dtype(meta["dtype"]))
    cache.mkdir(parents=True, exist_ok=True)
    for i, kz in enumerate(ks[0]):
        for j, ky in enumerate(ks[1]):
            for k, kx in enumerate(ks[2]):
                f = cache / f"{kz}.{ky}.{kx}"
                if not f.exists():
                    tmp = f.with_suffix(".part")
                    tmp.write_bytes(get(f"{source}/0/{kz}/{ky}/{kx}"))
                    tmp.rename(f)
                a = np.frombuffer(decode(f.read_bytes()), meta["dtype"]).reshape((c, c, c))
                buf[i * c : (i + 1) * c, j * c : (j + 1) * c, k * c : (k + 1) * c] = a
    s = np.asarray(lo) - base
    e = s + (np.asarray(hi) - np.asarray(lo))
    return buf[s[0] : e[0], s[1] : e[1], s[2] : e[2]]


def read_padded(cube: str, cache: Path, margin: int = MARGIN, get=_get, get_ranges=_get_ranges) -> np.ndarray:
    """The cube's padded region of its raw uint16 scan."""
    lo, hi = padded_bounds(cube, margin)
    scroll = cube.split("_")[0]
    if scroll in TIFF_VOLUMES:
        return read_tiff_region(TIFF_VOLUMES[scroll], lo, hi, cache, get_ranges)
    if scroll in ZARR_VOLUMES:
        return read_zarr_region(ZARR_VOLUMES[scroll], lo, hi, cache, get)
    raise ValueError(f"{cube}: no raw scan for {scroll}")


def load_cube_ct(path: Path) -> np.ndarray:
    """The cube's own CT: a .npy, or villa's multi-page uint8 .tif (read with Pillow)."""
    if path.suffix == ".npy":
        return np.load(path)
    from PIL import Image

    with Image.open(path) as im:
        pages = []
        for i in range(im.n_frames):
            im.seek(i)
            pages.append(np.array(im))
    return np.stack(pages)


def core(padded: np.ndarray, margin: int = MARGIN) -> np.ndarray:
    n = padded.shape[0] - 2 * margin
    return padded[margin : margin + n, margin : margin + n, margin : margin + n]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cube", help="the target's stem, e.g. s1_00497_01497_03997_256")
    ap.add_argument("--out", type=Path, required=True, help="the padded uint8 array, as .npy")
    ap.add_argument("--margin", type=int, default=MARGIN)
    ap.add_argument("--cache", type=Path, default=Path(".context_ct_cache"))
    ap.add_argument("--check", type=Path, help="the cube's own CT (.tif or .npy): its voxels must be the core")
    args = ap.parse_args(argv)
    padded = to_uint8(read_padded(args.cube, args.cache / args.cube, args.margin))
    if args.check is not None:
        ref = load_cube_ct(args.check)
        got = core(padded, args.margin)
        if got.shape != ref.shape or not np.array_equal(got, ref):
            n = int((got != ref).sum()) if got.shape == ref.shape else "shape"
            print(f"{args.cube}: the core DIFFERS from {args.check} ({n} voxels)")
            return 1
        print(f"{args.cube}: the core equals {args.check} exactly ({ref.size} voxels)")
    np.save(args.out, padded)
    print(f"wrote {args.out}: {padded.shape} uint8, the cube at [{args.margin}, {args.margin + padded.shape[0] - 2 * args.margin})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
