#!/usr/bin/env bash
# install.sh: bootstrap mpv + VapourSynth + Practical-RIFE for realtime
# frame interpolation on an Ubuntu/Debian machine (tested on Ubuntu 24.04,
# target hardware: 13th Gen Intel Core i3-1315U with its iGPU).
#
# Run from the repository root:  ./install.sh
# Stages are idempotent; re-running only rebuilds missing pieces.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${REPO_ROOT}/models"
ONNX_DIR="${REPO_ROOT}/dist"
VS_TAG="R66"
MPV_TAG="v0.40.0"
MPV_BUILD_OPTS="-Dvapoursynth=enabled"

sudo_missing() {
    if ! sudo -n true 2>/dev/null; then
        echo "This installer needs sudo (no password). Re-run inside a sudo session or configure NOPASSWD." >&2
        exit 1
    fi
}
sudo_missing

SUDO="sudo"

echo "==> [1/8] apt dependencies"
$SUDO apt-get update -qq
# enable deb-src for build-dep
if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then
    $SUDO sed -i 's/^Types: deb$/Types: deb deb-src/' /etc/apt/sources.list.d/ubuntu.sources
    $SUDO apt-get update -qq
fi
$SUDO apt-get install -y -qq build-essential autoconf automake libtool pkg-config \
    python3 python3-dev python3-pip python3-cython ninja-build meson \
    curl unzip ca-certificates git ffmpeg nasm yasm \
    libzimg-dev libluajit-5.1-dev 2>&1 | tail -n1
$SUDO apt-get build-dep -y -qq mpv 2>&1 | tail -n1

echo "==> [2/8] VapourSynth ${VS_TAG}"
if ! python3 -c "import vapoursynth; import sys; sys.exit(0 if 'Core R66' in str(vapoursynth.core) or True else 1)" 2>/dev/null; then
    if [ ! -d "${REPO_ROOT}/tmp/vapoursynth" ]; then
        git clone --depth 1 --branch ${VS_TAG} https://github.com/vapoursynth/vapoursynth "${REPO_ROOT}/tmp/vapoursynth"
    fi
    ( cd "${REPO_ROOT}/tmp/vapoursynth" &&
      ./autogen.sh &&
      ./configure --prefix=/usr/local &&
      make -j"$(nproc)" &&
      $SUDO make install &&
      $SUDO ldconfig )
    VSPKGDIR="$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib", "posix_local"))')"
    BINDIR="$(dirname "${VSPKGDIR}")"
    if [ -d "$(find /usr/local/lib -maxdepth 3 -name 'site-packages' -path '*python3*' | head -1)" ]; then
        SP="$(find /usr/local/lib -maxdepth 3 -name 'site-packages' -path '*python3*' | head -1)"
        $SUDO sh -c "echo '${SP}' > '${BINDIR}/mpv-practical-rife.pth'"
    fi
fi
python3 - <<'EOF'
import vapoursynth
print("VapourSynth:", vapoursynth.core.version() if callable(getattr(vapoursynth.core, "version", None)) else str(vapoursynth.core))
EOF

echo "==> [3/8] Python packages"
pip3 install --user -q --break-system-packages numpy onnxruntime gdown
python3 -c "import numpy" 2>/dev/null || true

if python3 -c "import openvino" 2>/dev/null; then
    echo "openvino present"
else
    pip3 install --user -q --break-system-packages openvino || echo "WARN: openvino install failed (iGPU acceleration unavailable; falling back to onnxruntime CPU)"
fi

echo "==> [4/8] Intel iGPU OpenCL/VA-API runtime (optional)"
if lspci 2>/dev/null | grep -qiE "intel.*(graphics|uhd|hd graphics)"; then
    $SUDO apt-get install -y -qq intel-media-va-driver-non-free clinfo 2>&1 | tail -n1 || true
    # OpenCL driver for the iGPU (needed by OpenVINO GPU plugin)
    if ! dpkg -l intel-compute-runtime >/dev/null 2>&1; then
        $SUDO mkdir -p /usr/share/keyrings
        curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key | \
            $SUDO tee /usr/share/keyrings/intel-gpg-keyring.gpg >/dev/null || true
        echo "deb [signed-by=/usr/share/keyrings/intel-gpg-keyring.gpg] https://repositories.intel.com/gpu/ubuntu noble unified" | \
            $SUDO tee /etc/apt/sources.list.d/intel-gpu.list >/dev/null || true
        $SUDO apt-get update -qq || true
        $SUDO apt-get install -y -qq intel-compute-runtime || echo "WARN: intel-compute-runtime failed; OpenVINO will run on CPU"
    fi
else
    echo "No Intel iGPU detected; skipping."
fi

echo "==> [5/8] mpv ${MPV_TAG} with vapoursynth"
if ! command -v mpv >/dev/null 2>&1 || ! mpv --vf=help 2>/dev/null | grep -q vapoursynth; then
    if [ ! -d "${REPO_ROOT}/tmp/mpv" ]; then
        git clone --depth 1 --branch ${MPV_TAG} https://github.com/mpv-player/mpv "${REPO_ROOT}/tmp/mpv"
    fi
    ( cd "${REPO_ROOT}/tmp/mpv" &&
      export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig &&
      ( meson setup build --prefix /usr/local --buildtype release ${MPV_BUILD_OPTS} || meson setup build --prefix /usr/local --buildtype release ${MPV_BUILD_OPTS} --wipe ) &&
      ninja -C build -j"$(nproc)" &&
      $SUDO ninja -C build install )
    $SUDO ldconfig
fi
mpv --vf=help 2>/dev/null | grep -q vapoursynth || { echo "FATAL: mpv lacks vapoursynth support" >&2; exit 1; }

echo "==> [6/8] Practical-RIFE models"
mkdir -p "${MODELS_DIR}"
download_model() {  # id, dest
    local id="$1" dest="$2"
    [ -f "${dest}/train_log/flownet.pkl" ] && return 0
    local zip="${REPO_ROOT}/tmp/$(basename "${dest}").zip"
    [ -f "${zip}" ] || python3 -m gdown --no-cookies "${id}" -O "${zip}"
    unzip -oq "${zip}" -d "${dest}"
}
download_model 1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg "${MODELS_DIR}/rife-4.25"
download_model 1zlKblGuKNatulJNFf5jdB-emp9AqGK05 "${MODELS_DIR}/rife-4.25-lite"

echo "==> [7/8] ONNX export"
if ! python3 -c "import torch" 2>/dev/null; then
    pip3 install --user -q --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu
fi
if ! python3 -c "import cv2" 2>/dev/null; then
    pip3 install --user -q --break-system-packages opencv-python-headless
fi
mkdir -p "${ONNX_DIR}"
[ -f "${REPO_ROOT}/tmp/Practical-RIFE/model/warplayer.py" ] || \
    git clone --depth 1 https://github.com/hzwer/Practical-RIFE "${REPO_ROOT}/tmp/Practical-RIFE"
python3 -m rife.export_ifnet "${MODELS_DIR}/rife-4.25/train_log" "${ONNX_DIR}/rife-4.25-s1.0.onnx"
python3 -m rife.export_ifnet "${MODELS_DIR}/rife-4.25-lite/train_log" "${ONNX_DIR}/rife-4.25-lite-s1.0.onnx"

echo "==> [8/8] mpv integration (scripts + keybinding)"
MPV_CONFIG="${HOME}/.config/mpv"
mkdir -p "${MPV_CONFIG}/scripts"
cp "${REPO_ROOT}/mpv/rife.lua" "${MPV_CONFIG}/scripts/rife.lua"
cp "${REPO_ROOT}/mpv/rife_video.py" "${MPV_CONFIG}/scripts/rife_video.py"
mkdir -p "${HOME}/.local/bin"
cp "${REPO_ROOT}/bin/rife-mpv" "${HOME}/.local/bin/rife-mpv"
chmod +x "${HOME}/.local/bin/rife-mpv"

cat <<'EOF'

Done. Try it:

    MPV_RIFE_MODEL=~/.local/share/... rife-mpv video.mkv
or simply:
    MPV_RIFE_MODEL=$(ls dist/rife-4.25-lite-s1.0.onnx) ./bin/rife-mpv video.mkv

Press Ctrl+r during playback to toggle RIFE interpolation on/off.
OpenVINO (Intel iGPU) is used automatically when installed; set
MPV_RIFE_BACKEND=ort to force onnxruntime CPU instead.
EOF
