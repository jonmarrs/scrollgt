"""tools/context_ct.py, offline: synthetic TIFF slices behind a fake range server, and a synthetic zarr."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import context_ct as cx  # noqa: E402


def _tiff(arr: np.ndarray, bits: int = 16, compression: int = 1, strips: int = 1) -> bytes:
    """A classic little-endian TIFF: one IFD, the image in STRIPS strips, as the raw scans are stored (one strip)."""
    h, w = arr.shape
    data = arr.astype("<u2" if bits == 16 else "u1").tobytes()
    n_entries = 9
    ifd_off = 8
    data_off = ifd_off + 2 + 12 * n_entries + 4
    per = len(data) // strips

    def entry(tag, typ, cnt, val):
        if typ == 3 and cnt == 1:
            return struct.pack("<HHIHH", tag, typ, cnt, val, 0)
        return struct.pack("<HHII", tag, typ, cnt, val)

    entries = [entry(256, 4, 1, w), entry(257, 4, 1, h), entry(258, 3, 1, bits), entry(259, 3, 1, compression),
               entry(262, 3, 1, 1), entry(273, 4, strips, data_off), entry(277, 3, 1, 1),
               entry(278, 4, 1, h // strips), entry(279, 4, strips, per)]  # fmt: skip
    head = b"II*\x00" + struct.pack("<I", ifd_off) + struct.pack("<H", n_entries) + b"".join(entries) + b"\x00" * 4
    return head + data


def _server(files: dict[str, bytes], multipart_from: int = 2):
    """get_ranges for FILES: one range -> a single-range 206, more -> multipart/byteranges, as nginx answers."""
    calls = []

    def get_ranges(url, ranges):
        calls.append((url, len(ranges)))
        data = files[url]
        ranges = [(a, min(b, len(data) - 1)) for a, b in ranges]  # clipped at the end, as HTTP servers do
        if len(ranges) < multipart_from:
            ((a, b),) = ranges
            return "image/tiff", f"bytes {a}-{b}/{len(data)}", data[a : b + 1]
        boundary = "0000000000000000042"
        parts = [f"\r\n--{boundary}\r\nContent-Type: image/tiff\r\nContent-Range: bytes {a}-{b}/{len(data)}\r\n\r\n".encode()
                 + data[a : b + 1] for a, b in ranges]  # fmt: skip
        return f"multipart/byteranges; boundary={boundary}", None, b"".join(parts) + f"\r\n--{boundary}--\r\n".encode()

    return get_ranges, calls


def test_the_header_is_parsed_and_checked():
    arr = np.arange(30 * 40, dtype="<u2").reshape(30, 40)
    info = cx.parse_header(_tiff(arr)[: cx.HEADER_BYTES])
    assert (info["width"], info["height"]) == (40, 30)
    with pytest.raises(ValueError, match="bits 8"):
        cx.parse_header(_tiff(arr.astype(np.uint8), bits=8)[: cx.HEADER_BYTES])
    with pytest.raises(ValueError, match="compression 8"):
        cx.parse_header(_tiff(arr, compression=8)[: cx.HEADER_BYTES])
    with pytest.raises(ValueError, match="3 strips"):
        cx.parse_header(_tiff(arr, strips=3)[: cx.HEADER_BYTES])
    with pytest.raises(ValueError):
        cx.parse_header(b"MM\x00*" + bytes(100))  # big-endian


def test_a_slice_window_is_exact_in_single_and_multipart_responses():
    arr = np.random.default_rng(3).integers(0, 65535, (50, 70), dtype=np.uint16)
    files = {"u/0.tif": _tiff(arr)}
    for per in (1, 7, 200):
        get, calls = _server(files)
        win = cx.read_slice_window("u/0.tif", (5, 11), (33, 60), get, per_request=per)
        assert np.array_equal(win, arr[5:33, 11:60])
        assert max(n for _, n in calls) <= per and calls[0] == ("u/0.tif", 1)
    with pytest.raises(ValueError):
        cx.read_slice_window("u/0.tif", (0, 0), (51, 10), _server(files)[0])
    with pytest.raises(ValueError):
        cx.parse_byteranges(b"abc", "image/tiff", "bytes 0-9/100")


def test_a_tiff_region_stacks_slices_and_caches_them(tmp_path):
    vol = np.random.default_rng(4).integers(0, 65535, (6, 20, 30), dtype=np.uint16)
    files = {f"base/{z:04d}.tif": _tiff(vol[z]) for z in range(6)}
    volume = {"url": "base", "shape": (6, 20, 30), "digits": 4}
    get, calls = _server(files)
    got = cx.read_tiff_region(volume, (1, 2, 3), (5, 19, 29), tmp_path, get)
    assert np.array_equal(got, vol[1:5, 2:19, 3:29])
    n = len(calls)
    assert np.array_equal(cx.read_tiff_region(volume, (1, 2, 3), (5, 19, 29), tmp_path, get), got)
    assert len(calls) == n  # every slice from the cache
    with pytest.raises(ValueError):
        cx.read_tiff_region(volume, (0, 0, 0), (7, 1, 1), tmp_path, get)


def _fake_zarr(shape=(12, 12, 12), chunk=4):
    vol = np.random.default_rng(1).integers(0, 65535, shape, dtype=np.uint16)
    meta = {"chunks": [chunk] * 3, "compressor": None, "dimension_separator": "/", "dtype": "<u2",
            "shape": list(shape), "zarr_format": 2}  # fmt: skip
    calls = []

    def get(url: str) -> bytes:
        calls.append(url)
        if url.endswith("/0/.zarray"):
            return json.dumps(meta).encode()
        kz, ky, kx = (int(v) for v in url.removeprefix("fake/0/").split("/"))
        c = chunk
        return np.ascontiguousarray(vol[kz * c : (kz + 1) * c, ky * c : (ky + 1) * c, kx * c : (kx + 1) * c]).tobytes()

    return vol, get, calls


def test_a_zarr_region_is_exact_and_caches_chunks(tmp_path):
    vol, get, calls = _fake_zarr()
    got = cx.read_zarr_region("fake", (1, 3, 5), (9, 10, 12), tmp_path, get)
    assert np.array_equal(got, vol[1:9, 3:10, 5:12])
    n = len(calls)
    assert np.array_equal(cx.read_zarr_region("fake", (1, 3, 5), (9, 10, 12), tmp_path, get), got)
    assert len(calls) == n + 1  # only the .zarray again: every chunk came from the cache
    with pytest.raises(ValueError):
        cx.read_zarr_region("fake", (0, 0, 0), (13, 4, 4), tmp_path, get)


def test_the_conversion_is_villas_generator():
    raw = np.array([[[0, 1, 256, 257, 514, 32767, 65278, 65279, 65535]]], np.uint16)
    assert cx.to_uint8(raw).ravel().tolist() == [0, 0, 0, 1, 2, 127, 254, 254, 255]
    big = np.random.default_rng(5).integers(0, 65535, (150, 3, 4), dtype=np.uint16)
    want = np.clip(big.astype(np.float64) / 257, 0, 255).astype(np.uint8)
    assert np.array_equal(cx.to_uint8(big), want) and np.array_equal(cx.to_uint8(big, slab=7), want)


def test_each_scroll_is_read_from_its_own_scan(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cx, "read_tiff_region", lambda vol, lo, hi, cache, get_ranges: calls.append((vol["url"], lo, hi)))
    monkeypatch.setattr(cx, "read_zarr_region", lambda src, lo, hi, cache, get: calls.append((src, lo, hi)))
    for cube in ("s1_00497_01497_03997_256", "s3_00994_00494_01994_512", "s5_07997_02997_05497_256"):
        cx.read_padded(cube, tmp_path)
    assert calls == [
        (cx.TIFF_VOLUMES["s1"]["url"], (369, 1369, 3869), (881, 1881, 4381)),
        (cx.TIFF_VOLUMES["s3"]["url"], (866, 366, 1866), (1634, 1134, 2634)),
        (cx.ZARR_VOLUMES["s5"], (7869, 2869, 5369), (8381, 3381, 5881)),
    ]
    with pytest.raises(ValueError, match="no raw scan"):
        cx.read_padded("s4_00000_00000_00000_256", tmp_path)


def test_every_target_lies_inside_its_scan_with_the_margin():
    shapes = {**{k: v["shape"] for k, v in cx.TIFF_VOLUMES.items()}, "s5": (21000, 6700, 9100)}  # s5: its level 0
    cubes = [p.name.removeprefix("fibers_") for p in (ROOT / "data").glob("fibers_*")]
    assert len(cubes) == 21
    for c in cubes:
        lo, hi = cx.padded_bounds(c)
        assert all(a >= 0 for a in lo) and all(b <= s for b, s in zip(hi, shapes[c[:2]], strict=True)), c


def test_the_scans_are_the_ones_the_cubes_were_cut_from():
    assert cx.TIFF_VOLUMES["s1"]["url"].endswith("Scroll1/PHercParis4.volpkg/volumes/20230205180739")
    assert cx.TIFF_VOLUMES["s3"]["url"].endswith("Scroll3/PHerc332.volpkg/volumes/20231117143551")
    assert cx.ZARR_VOLUMES["s5"].endswith("Scroll5/PHerc172.volpkg/volumes_zarr/20241024131838.zarr")
    assert cx.MARGIN == 128


def _run_main(tmp_path, monkeypatch, padded, ref, suffix=".npy"):
    monkeypatch.setattr(cx, "read_padded", lambda cube, cache, margin: padded)
    path = tmp_path / f"cube{suffix}"
    if suffix == ".npy":
        np.save(path, ref)
    else:
        from PIL import Image

        pages = [Image.fromarray(p) for p in ref]
        pages[0].save(path, save_all=True, append_images=pages[1:])
    out = tmp_path / "padded.npy"
    rc = cx.main(["s1_00000_00000_00000_4", "--out", str(out), "--margin", "2", "--check", str(path),
                  "--cache", str(tmp_path / "cache")])  # fmt: skip
    return rc, out


def test_check_requires_the_core_to_be_the_cube(tmp_path, monkeypatch):
    raw = np.random.default_rng(6).integers(0, 65535, (8, 8, 8), dtype=np.uint16)
    ref = cx.to_uint8(raw)[2:6, 2:6, 2:6]
    for suffix in (".npy", ".tif"):
        rc, out = _run_main(tmp_path, monkeypatch, raw, ref, suffix)
        assert rc == 0 and np.array_equal(np.load(out)[2:6, 2:6, 2:6], ref) and np.load(out).shape == (8, 8, 8)
        out.unlink()
    bad = ref.copy()
    bad[1, 1, 1] ^= 1
    rc, out = _run_main(tmp_path, monkeypatch, raw, bad)
    assert rc == 1 and not out.exists()


def test_a_failed_request_is_tried_again(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def flaky(x):
        calls.append(x)
        if len(calls) < 3:
            raise TimeoutError("The read operation timed out")
        return x * 2

    assert cx.retry(flaky, 21) == 42 and len(calls) == 3
    calls.clear()
    with pytest.raises(TimeoutError):
        cx.retry(flaky, 1, tries=2)


README = (ROOT / "README.md").read_text()
CUBE_CT = "https://dl.ash2txt.org/datasets/fiber-skeletons/Dataset004_sk-fibers-binary-20250728/imagesTr/"


def _bash_commands(text: str) -> list[str]:
    """Every command line in the README's ```bash blocks (comments and blank lines left out)."""
    blocks = text.split("```bash\n")[1:]
    lines = [line.strip() for b in blocks for line in b.split("```", 1)[0].splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def test_the_readme_runs_the_tool_as_its_quickstart_installs_it():
    """Run from a fresh clone, the README's `python tools/context_ct.py ...` exited 127: stock Ubuntu has no `python`,
    and the quickstart installs into .venv, whose interpreter is .venv/bin/python."""
    commands = _bash_commands(README)
    assert not [c for c in commands if c.split()[0] == "python"]
    tool = [c for c in commands if "tools/context_ct.py" in c]
    assert tool and all(c.startswith(".venv/bin/python tools/context_ct.py ") for c in tool)
    assert ".venv/bin/pip install numcodecs" in commands


def test_the_readme_says_where_the_cube_ct_for_check_is():
    """--check needs the cube's own CT; it is public, beside the ground truth every target names."""
    datasets = {json.loads(m.read_text())["ground_truth"]["nml"]["dataset"] for m in (ROOT / "data").glob("fibers_*/meta.json")}
    assert datasets == {CUBE_CT.split("/")[-3]}
    stem = "s1_00497_01497_03997_256"
    assert f"curl -O {CUBE_CT}{stem}_0000.tif" in _bash_commands(README)
    assert f".venv/bin/python tools/context_ct.py {stem} --out context.npy --check {stem}_0000.tif" in _bash_commands(README)
