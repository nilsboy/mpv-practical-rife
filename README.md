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

* Linux. Ubuntu 24.04 was used for development.
* `mpv` compiled with `--enable-vapoursynth` (Ubuntu's stock mpv does
  **not** ship it; use `./install.sh`, which builds VapourSynth R66 + mpv
  0.40.0 with the filter enabled, or a distro build that has it).
* VapourSynth R66 (core + python bindings).
* Python packages: `numpy`, `onnxruntime`, `openvino` (for the iGPU).
* A Practical-RIFE release (weights + architecture from
  [the model list](https://github.com/hzwer/Practical-RIFE#trained-model))
  and `rife/export_ifnet.py` to convert it to ONNX (needs `torch` on CPU).

## Quick start (development repo)

```sh
./install.sh                        # full bootstrap (apt, builds, models)
MPV_RIFE_MODEL=dist/rife-4.25-lite-s1.0.onnx ./bin/rife-mpv video.mkv
```

During playback press **Ctrl+r** to toggle interpolation.

## Snap

A self-contained classic snap is packaged with `snapcraft.yaml` (base
core24, classic confinement). It bundles its own mpv + VapourSynth build,
the Python runtimes (numpy, onnxruntime, **OpenVINO** for Intel iGPU) and
both RIFE models, so it needs no host setup beyond snapd:

```sh
snapcraft pack                      # or use dist/mpv-practical-rife_0.1.0_amd64.snap
sudo snap install --dangerous --classic ./dist/mpv-practical-rife_*.snap

mpv-practical-rife.rife-mpv video.mkv         # Ctrl+r toggles interpolation
mpv-practical-rife.bench --synthetic 1274x714
```

Classic confinement is required for the embedded-Python VapourSynth
pipeline and the host's iGPU OpenCL/VA-API drivers (strict confinement has
no clean OpenCL story). The snap still exposes all `MPV_RIFE_*` overrides,
e.g. `MPV_RIFE_MODEL=/path/model.onnx mpv-practical-rife.rife-mpv ...`; the lua
toggle script reads its config through `--script-opts`/config as usual.

## Usage

```sh
rife-mpv [--script-opts rife-<opt>=<value> ...] file...
```

The wrapper starts playback with RIFE enabled (`Ctrl+r` still toggles it
off/on). Without mpv scripting, use the filter directly:

```sh
mpv --vf='vapoursynth=~/.config/mpv/scripts/rife_video.py:buffered-frames=10:concurrent-frames=2' video.mkv
```

### Settings

Precedence: script-opts/Lua options > `user-data` JSON > `MPV_RIFE_*` env
> defaults. Long name or short key both work.

| Env / option      | Key | Default | Meaning |
|-------------------|-----|---------|---------|
| `MPV_RIFE_MODEL`  | `m` | — | path to the ONNX model (required) |
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

`rife.lua` (in `~/.config/mpv/scripts/`) reads the same settings from
`~/.config/mpv/mpv.conf` under the script section name, e.g.
`script-opts=rife-model=/abs/path/rife.onnx:...`, or pass them with
`--script-opts=rife-factor_num=4`.

### Exporting other models

```sh
python3 -m rife.export_ifnet <train_log_dir> out.onnx [--scale 0.5]
```

`<train_log_dir>` is the unpacked Practical-RIFE release (contains
`flownet.pkl`, `RIFE_HDv3.py`, `IFNet_HDv3.py`). `--scale` bakes the optical
flow resolution factor into the graph (0.5 halves compute; hzwer recommends
it for 4K).

### Benchmarking

```sh
python3 tools/bench.py --model dist/rife-4.25-lite-s1.0.onnx --synthetic 1274x714
```

prints interpolated frames/sec and whether realtime (factor × source fps)
is met for the hardware at hand.

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

    rife/          python package (config, backends, VS filter, ONNX export)
    mpv/           rife_video.py (vf_vapoursynth entry) + rife.lua (toggle)
    bin/rife-mpv     launcher
    tools/bench.py synthetic-source benchmark
    tests/         config/vsfilter/backends tests + ONNX↔torch parity test
    models/        downloaded Practical-RIFE releases (gitignored)
    snap/          snap packaging (models, wrappers)
    snapcraft.yaml classic-snap definition
    dist/          final artifacts: mpv binary + ONNX + snap (gitignored)
    install.sh     target-machine bootstrap
    NOTES.md       development notes and implementation details

## Tests

```sh
python3 -m unittest discover -s tests
```

Runs 15 tests; the export-parity test needs torch + model weights and is
skipped otherwise.
