# NOTES

Development notes, decisions and gotchas discovered while building
mpv-practical-rife.

The repository is snap-only: `snapcraft pack` is the one build step, the
snap is the only deliverable, and the released `.snap` lives in `dist/`
(the earlier source-build `install.sh` and standalone mpv/ONNX toolchain
were trimmed; both are recoverable from git history).

## Architecture decisions

* **mpv `vf_vapoursynth` is the only sane runtime integration.** mpv cannot
  inject frames from the outside; the vapoursynth bridge gives a proper
  push/pull frame interface with per-frame timing properties.
* **vs-rife (HolyWu) was rejected**: v5.7.0 is CUDA-torch only, useless for
  the Intel iGPU target. vs-mlrt would work but ships no Linux
  CPU/OpenVINO plugin builds from its releases; building the plugin chain
  (onnxruntime C lib + protobuf + onnx) is heavy. Instead the VS filter is
  implemented in pure Python (`rife/vsfilter.py`) on top of pip-installed
  `onnxruntime`/`openvino` — zero compiled plugins.
* **ONNX export — historical**: the bundled `snap/models/*.onnx` were
  produced by a torch-based exporter (`rife/export_ifnet.py`, removed from
  the repo after the snap-only trim; see git history) that re-implemented
  the v4.x inference graph with the same weights, dropping the flow/mask
  outputs. The author's own forward cannot be traced because it builds a
  python list that mixes `(warped0, warped1)` tuples and later a tensor
  (`merged[4] = ...`), which the TorchScript tracer rejects with "List
  inputs ... must have consistent element type". Parity with the original
  was verified (PSNR ≈ 109 dB against `RIFE_HDv3.Model.inference`).
* **Warp grid is rebuilt per call** (no python dict caching as in
  warplayer.py) so the traced graph keeps `aten::size→linspace`
  dependencies and stays correct for dynamic input sizes.
* **The new torch.export-based ONNX exporter (torch ≥ 2.9 default) fails**
  on the symbolic shape guards inside the warp math ("cannot determine
  truth value of Relational: 0 < (((-1.0)//2)) + 1"); `dynamo=False` (the
  TorchScript tracer) works.
* **Input padding to 128** (hzwer's own `inference_video.py` pads H/W to
  `max(128, 128/scale)`): the IFBlock scale chain only survives
  ceil-arithmetic for those multiples. Zero padding (constant 0), matching
  `F.pad` defaults.

## mpv ↔ VapourSynth contract (mpv 0.40 `vf_vapoursynth.c`)

* Script receives globals: `video_in` (VS node), `video_in_dw/dh`,
  `container_fps` (float), `display_fps`, `display_res`, `user_data`.
* Output must be registered with `clip.set_output(0)` (`_vpy_evaluate`
  reads output 0).
* mpv only accepts **planar YUV** in *both* directions
  (`get_valid_mp_regular_imgfmt` requires 3 planes + cfYUV
  round-trips in `mp_to_vs`/`mp_from_vs`). RGBS is therefore converted
  back to the input YUV format by the filter.
* Every output frame **must** carry `_DurationNum/_DurationDen` frame
  properties or mpv logs "No PTS after filter" and pacing breaks.
* `buffered-frames` (default 4) × `concurrent-frames` (default #CPU) is the
  input window; requesting sources older than the window fails ("Frame N
  requested, but only have frames starting from M" — our filter only needs
  the previous frame pair, so a window of 8–32 is comfortable with slow
  inference).
* Filter suboptions use dash-separated names; underscores are rejected.
  Suboption values cannot contain `:`, `=`, or `,` — hence the base64 JSON
  payload for `user-data` (mpv/rife.lua emits padding-free base64 because
  `=` also breaks suboption parsing).

## Frames-derivation gotcha (crash on filter teardown)

Returning VapourSynth frames whose source was a `std.BlankClip` out of a
multi-input `std.ModifyFrame` inside `std.FrameEval` aborts mpv at teardown
("terminate called without an active exception") — even when the filter is
otherwise healthy. Passing through copies of *real* source frames and
overwriting their pixels works reliably (this is what
`rife/vsfilter.py::_InterpState` does). Verified by bisecting against
minimal scripts (`tmp/*_test.py` probes during development).

## CI (build-snap.yml) networking gotcha

* On GitHub-hosted runners `snapcraft pack` can fail while launching the
  LXD build instance with "A network related operation failed in a context
  of no network access." Two independent causes, both fixed in the install
  step, both after `lxd init --auto` (which creates lxdbr0):
  - Docker is preinstalled and sets the iptables FORWARD policy to DROP,
    which swallows all traffic forwarded by lxdbr0 → `sudo iptables -P
    FORWARD ACCEPT`.
  - The runner hands the instance a non-routable IPv6 address → `sudo lxc
    network set lxdbr0 ipv6.address none`.
* The failure surfaces during "Creating new base instance from remote" and
  misleadingly implicates the runner's own connectivity, not the
  container's (same workaround is used, e.g., by canonical/jaime CI).
* `sudo lxd waitready --timeout=60` before init guards against the just-
  installed lxd snap daemon not being up yet.

## Sandbox limitations (dev box)

* No Intel iGPU here (Red Hat Virtio GPU), so the OpenVINO **GPU** path is
  implemented but untested in this environment; OpenVINO-CPU and
  onnxruntime-CPU paths are fully exercised.
* Raw CPU interpolation on the 4-thread VM: ≈0.2 interp/s for 4.25 at
  720p+pad. The i3-1315U is substantially faster and the iGPU path is the
  intended realtime one; use `mpv-practical-rife.bench` on the target to
  find the workable resolution/model/scale.

## Benchmarks (sandbox, 4 threads)

```
4.25       1280x768:  6761.8 ms/frame  ( 0.15 interpolated fps)
4.25       1920x896: 11675.7 ms/frame  ( 0.09 interpolated fps)
4.25-lite  1280x768:  5417.3 ms/frame  ( 0.18 interpolated fps)
4.25-lite  1920x896:  9116.9 ms/frame  ( 0.11 interpolated fps)
```

## Model version notes

* 4.25 scale_list `[16,8,4,2,1]`; 4.25.lite `[32,16,8,4,1]` and a smaller
  block4 (c=24) — both parsed from the release's own `RIFE_HDv3.py`.
* Timestep is exported as a `[1,1,1,1]` tensor input; the VS filter runs it
  at `k/factor_num`.

## mpv Lua quirks (mpv 0.40, lua 5.1)

* No `bit` library, no `>>`/`&` operators: base64 written with math only.
* `mp.get_script_directory()` returns nil for scripts loaded via `--script`;
  the snap's lua falls back to `$SNAP/mpv/rife_video.py`, which resolves.
* `vf remove` matches **labels** (`@rife`), so the filter is added as
  `@rife:vapoursynth=...`
* script-opts keys are prefixed with the script name automatically.

## Snap packaging (snapcraft 9.2, core24, classic)

* **Python plugin unusable for classic here**: it insists on *provisioning*
  the interpreter into the payload (searches install/stage dirs during the
  build step, when classic snaps have no staged python yet) and aborts with
  "No suitable Python interpreter found". Since we only need importable
  libraries (numpy/onnxruntime/openvino) next to the VapourSynth bindings,
  the part installs them with `pip3 install --target` into
  `$SNAP/lib/python3.12/site-packages` instead.
* **VS python binding placement**: the autotools part installs it to
  `/usr/local/lib/python3.12/site-packages`; an `override-build` moves it
  to `$SNAP/lib/python3.12/site-packages` so both runtimes share one dir
  exposed through PYTHONPATH by the wrappers.
* **libvapoursynth lives in `$SNAP/usr/local/lib`** (VS built with
  `--prefix=/usr/local`); wrappers export `LD_LIBRARY_PATH` — snapd does not
  put `usr/local` on the search path by itself.
* No external VapourSynth plugin dir is needed: std/resize/text are compiled
  into the core (bound="Core", absent plugin dirs only log a warning).
* mpv executable path: craft's meson staging is inconsistent about bindir,
  so the part `override-build` finds the binary via `find` and moves it to
  `$SNAP/usr/bin/mpv`.
* `dump` parts stage *contents* at the snap root: `organize` keys are needed
  to nest `models/` and `wrappers/`; `organize` also runs *after*
  `override-build`, so chmods act on pre-organize paths (use `find`).
* snapcraft copies the whole project into the LXD build instance;
  stray files (sockets!) break the copy — `.gitignore`-excluded dirs are
  honored for local packing but the lxd transport takes everything, so the
  project must be clean before packing (`tmp/*.sock` bit us twice).
* Resulting snap ≈133 MB compressed (mpv+VS ~15 MB, python wheels ~500 MB
  unpacked, OpenVINO dominates).
* Verified on the target-like sandbox: `mpv-practical-rife.bench` end-to-end plus
  interpolated playback with stable on/off/on toggle cycles, all through
  `/snap/mpv-practical-rife/current` paths.
