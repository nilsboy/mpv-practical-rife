# mpv-practical-rife

Realtime [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) frame
interpolation inside [mpv](https://mpv.io), through mpv's
`vf_vapoursynth` bridge and a pure-Python VapourSynth filter that runs the
network with [onnxruntime](https://onnxruntime.ai) (CPU/CUDA) or
[OpenVINO](https://github.com/openvinotoolkit/openvino) (Intel iGPU/CPU).

Target hardware: 13th Gen Intel Core i3-1315U (4 P/E cores, Intel UHD iGPU).
The iGPU via OpenVINO is the intended realtime path; onnxruntime-on-CPU is
the portable fallback.

```
mpv ──vf_vapoursynth── rife_video.py (VSScript)
   video_in (YUV) ──► resize→RGBS ──► RIFE 2x (ONNX: ort/ov) ──► resize→YUV ──► clip
   per-frame _DurationNum/_DurationDen props drive mpv pacing
```

## Requirements

* Linux with **snapd** (the `ubuntu-24.04` CI builds on it). Everything else
  mpv, VapourSynth, numpy/onnxruntime/OpenVINO, the RIFE models — is inside
  the snap.
* For hacking on the Python filter code: a system VapourSynth (R66+) plus
  `numpy`/`onnxruntime` for the test suite.

## Quick start

```sh
sudo snap install --dangerous --classic mpv-practical-rife_0.1.0_amd64.snap
mpv-practical-rife.rife-mpv video.mkv
```

During playback press **Ctrl+r** to toggle interpolation. Releases carry the
snap automatically (see CI below); `snapcraft pack` builds one locally.

## Usage

```sh
mpv-practical-rife.rife-mpv [--script-opts rife-<opt>=<value> ...] file...
```

The wrapper starts playback with RIFE enabled (`Ctrl+r` still toggles it
off/on). Classic confinement gives it full file/GPU access, and the lua
toggle script reads its settings from `--script-opts`/config as usual.

### Settings

Precedence: script-opts/Lua options > `user-data` JSON > `MPV_RIFE_*` env
> defaults. Long name or short key both work. The model defaults to the
snap-bundled `rife-4.25-lite-s1.0.onnx`; override with:

```sh
MPV_RIFE_MODEL=/path/to/model.onnx mpv-practical-rife.rife-mpv video.mkv
```

| Env / option      | Key | Default | Meaning |
|-------------------|-----|---------|---------|
| `MPV_RIFE_MODEL`  | `m` | bundled 4.25.lite | path to the ONNX model |
| `MPV_RIFE_BACKEND`| `b` | `auto` | `auto` / `ort` / `ov` |
| `MPV_RIFE_DEVICE` | `d` | auto | e.g. `GPU`, `CPU`, `CUDA` |
| `MPV_RIFE_THREADS`| `t` | 0 (auto) | intra-op threads for CPU runtimes |
| `MPV_RIFE_FACTOR_NUM`| `f` | 2 | target fps multiplier numerator |
| `MPV_RIFE_FACTOR_DEN`| `g` | 1 | target fps multiplier denominator |
| `MPV_RIFE_SC`     | `s` | 0.15 | scene-change hold threshold (0 = off) |
| `MPV_RIFE_PAD`    | `p` | 128 | network input padding multiple |
| `MPV_RIFE_MATRIX` | `x` | 709 | YUV→RGB matrix (`709`, `470bg`, `170m`, `2020`) |
| `MPV_RIFE_VS_THREADS` | `v` | 2 | VapourSynth core threads |
| `MPV_RIFE_FPS_NUM/DEN` | – | container | source fps override |

Lua script options additionally: `autostart`, `buffered_frames`,
`concurrent_frames`, `script_path`, `vpy`.

### Benchmarking

```sh
mpv-practical-rife.bench --synthetic 1274x714
```

prints interpolated frames/sec and whether realtime (factor × source fps)
is met for the hardware at hand. `--model /path/model.onnx` overrides the
bundled model.

### Performance notes (i3-1315U)

* Prefer the **lite** models (`4.25.lite`) and the OpenVINO iGPU backend
  (`backend=auto` picks it when installed).
* 1080p realtime interpolation is not reachable on CPU with this class of
  hardware; 720p with lite models on the iGPU is the realistic target.
* When inference is slower than realtime, mpv keeps audio sync by dropping
  output frames — the picture becomes choppier rather than lagging.
* `--untimed` bypasses pacing (offline-quality marathons / benchmarking).

## CI

`.github/workflows/build-snap.yml` packs the snap on every push/PR
(`ubuntu-24.04` runner, LXD-backed `snapcraft pack`), smoke-tests it headless
(`bench` interpolation + `snap list`), and uploads the `.snap` as a workflow
artifact.

Every **push** publishes a GitHub release (no secrets needed — the default
`GITHUB_TOKEN` covers it):

* push to `master` → replaces the rolling **`latest` prerelease** snapshot
  with the freshly built snap (tag recreated at the pushed commit)
* push of a `v*` tag → creates an **exact release** with generated notes
  (`gh release upload --clobber` on re-pushed tags)
* PR and manual builds → artifact only, no release

To publish a version: bump `version:` in `snapcraft.yaml`, commit, and push
its tag:

```sh
git tag v0.1.0 && git push origin v0.1.0
```

## Layout

    rife/          python package (config, backends, VS filter)
    mpv/           rife_video.py (vf_vapoursynth entry) + rife.lua (toggle)
    snap/          snap packaging (bundled models + entry wrappers)
    snapcraft.yaml classic-snap definition
    tools/bench.py synthetic-source benchmark
    tests/         config/vsfilter tests + workflow contract tests
    dist/          final artifact: the built snap (gitignored)
    NOTES.md       development notes and implementation details

## Tests

```sh
python3 -m unittest discover -s tests
```

Requires a system VapourSynth plus numpy/onnxruntime; the filter tests run
against a synthetic blend model, so no RIFE weights are needed.
