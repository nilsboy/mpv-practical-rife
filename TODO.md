- [x] integrate mpv with https://github.com/hzwer/Practical-RIFE
- [x] optimize for real-time playback on a 13th Gen Intel® Core™ i3-1315U × 8
- [x] utilize the gpu

## New TODOs

- [x] package the whole stack as a classic core24 snap (snapcraft.yaml +
      snap/wrappers; bundled mpv, VapourSynth, onnxruntime, OpenVINO,
      both models) — packed at dist/mpv-practical-rife_0.1.0_amd64.snap and verified
      headless with toggle cycles
- [ ] measure ONNX latency on the real i3-1315U (iGPU via OpenVINO GPU, CPU
      via onnxruntime) with tools/bench.py and pick default
      model/scale/backend accordingly
- [ ] consider FP16 (OpenVINO GPU) or quantized exports for extra headroom
- [ ] adaptive quality: drop to scale 0.5 model when mpv reports sustained
      frame drops during playback
- [ ] optional offline interpolation CLI (rife-cli: input video -> 2x file)
      for pre-rendering content that realtime playback cannot handle
- [ ] snap store path: classic-use request for "mpv-practical-rife", or explore
      strict confinement (needs a usable iGPU OpenCL story)
- [ ] shrink snap: drop OpenVINO (~+150MB) into an optional second snap or
      accept host-provided openvino when present
