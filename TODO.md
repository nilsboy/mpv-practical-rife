- [x] integrate mpv with https://github.com/hzwer/Practical-RIFE
- [x] optimize for real-time playback on a 13th Gen Intel® Core™ i3-1315U × 8
- [x] utilize the gpu

## New TODOs

- [x] package the whole stack as a classic core24 snap (snapcraft.yaml +
      snap/wrappers; bundled mpv, VapourSynth, onnxruntime, OpenVINO,
      both models) — packed at dist/mpv-practical-rife_0.1.0_amd64.snap and verified
      headless with toggle cycles
- [x] add a GitHub Action building the snap (build-snap.yml: checkout,
      snapcraft+LXD install, pack, headless bench smoke test, artifact
      upload, validated with actionlint+shellcheck and contract tests)
- [x] publish releases from the workflow: `release` job (needs: build)
      publishes on EVERY push to master as a rolling `latest` prerelease
      snapshot (delete tag+release, recreate at pushed commit), and as an
      exact release with generated notes on v* tags (--clobber fallback);
      bin/upload-release removed; publish script covered by stubbed-gh tests
- [x] trim the repo to snap-only: removed install.sh (system mpv/VapourSynth
      bootstrap), bin/rife-mpv launcher, the torch ONNX exporter
      (rife/export_ifnet.py, recoverable from git history along with the
      raw weight downloads) and the parity test; dist/ now holds only the
      snap; snap remains self-contained (models bundled in snap/models)
- [x] fix build-snap.yml LXD networking on GitHub runners (pack died at
      "Creating new base instance from remote" with "A network related
      operation failed in a context of no network access"): after
      `lxd init --auto` add `iptables -P FORWARD ACCEPT` (preinstalled
      Docker sets FORWARD policy to DROP) and `lxc network set lxdbr0
      ipv6.address none` (non-routable IPv6); + lxd waitready; contract
      test added, details in NOTES.md
- [x] fix CI smoke test failing on `import vapoursynth`
      (libzimg.so.2 missing from the snap on clean hosts: stage
      `libzimg2` into the vapoursynth part and prepend the staged lib
      dirs in both wrappers' LD_LIBRARY_PATH; test_snap.py covers it)
- [ ] stage the mpv/ffmpeg runtime libraries into the snap (stage-packages:
      libplacebo/ass/ffmpeg/lua/etc.) so `rife-mpv` works on clean Ubuntu
      24.04 hosts; today classic only resolves them if installed there
- [ ] measure ONNX latency on the real i3-1315U (iGPU via OpenVINO GPU, CPU
      via onnxruntime) with mpv-practical-rife.bench and pick default
      model/scale/backend accordingly
- [ ] consider FP16 (OpenVINO GPU) or quantized exports for extra headroom
      (would require re-adding an exporter from git history)
- [ ] adaptive quality: drop to scale 0.5 model when mpv reports sustained
      frame drops during playback
- [ ] optional offline interpolation CLI (rife-cli: input video -> 2x file)
      for pre-rendering content that realtime playback cannot handle
- [ ] snap store path: classic-use request for "mpv-practical-rife", or explore
      strict confinement (needs a usable iGPU OpenCL story)
- [ ] shrink snap: drop OpenVINO (~+150MB) into an optional second snap or
      accept host-provided openvino when present
