"""Export Practical-RIFE (HzWER) models to ONNX.

Loads the model author's own architecture/weights from ``train_log/``
(RIFE_HDv3.py + IFNet_HDv3.py + flownet.pkl, as distributed together with each
model release on https://github.com/hzwer/Practical-RIFE), wraps it so that the
network takes ``x  [B,6,H,W]`` (image0 RGB on top of image1 RGB, scaled to
[0,1]) plus a scalar ``timestep`` tensor and returns the merged middle frame
``[B,3,H,W]``, then traces that to ONNX.

Only the final merged frame is exported; the flow/mask outputs that the
original forward returns are dropped.
"""

from __future__ import annotations

import os
import shutil
import sys

import torch
from torch import nn


def _find_practical_rife_root() -> str:
    env_root = os.environ.get("PRACTICAL_RIFE_ROOT")
    if env_root and os.path.isdir(os.path.join(env_root, "model")):
        return os.path.abspath(env_root)
    candidates = []
    here = os.path.dirname(os.path.abspath(__file__))
    for parent in (here, os.path.join(here, "..")):
        candidates.append(parent)
        candidates.append(os.path.join(parent, "tmp", "Practical-RIFE"))
    for cand in candidates:
        if os.path.exists(os.path.join(cand, "model", "warplayer.py")):
            return os.path.abspath(cand)
    raise RuntimeError(
        "Cannot locate Practical-RIFE checkout (need model/warplayer.py). "
        "git clone https://github.com/hzwer/Practical-RIFE and point "
        "PRACTICAL_RIFE_ROOT at it."
    )


def _prepare_import_env(model_dir: str, workdir: str) -> str:
    """Build a directory so that ``RIFE_HDv3``'s imports resolve.

    RIFE_HDv3.py (shipped in the model zip) does
    ``from train_log.IFNet_HDv3 import *`` and
    ``from model.warplayer import warp``, i.e. it expects to be imported from
    the repo root with the model weights unpacked into ``train_log/``.
    """
    env_dir = os.path.join(workdir, "import_env")
    os.makedirs(os.path.join(env_dir, "model"), exist_ok=True)

    root = _find_practical_rife_root()
    for name in ("warplayer.py", "loss.py"):
        src = os.path.join(root, "model", name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(env_dir, "model", name))
    open(os.path.join(env_dir, "model", "__init__.py"), "w").close()

    msssim_src = os.path.join(root, "model", "pytorch_msssim")
    msssim_dst = os.path.join(env_dir, "model", "pytorch_msssim")
    if os.path.isdir(msssim_src) and not os.path.isdir(msssim_dst):
        shutil.copytree(msssim_src, msssim_dst)

    link = os.path.join(env_dir, "train_log")
    if not os.path.exists(link):
        os.symlink(os.path.abspath(model_dir), link)
    open(os.path.join(os.path.abspath(model_dir), "__init__.py"), "a").close()

    return env_dir


def _parse_scale_list(model_dir: str) -> list[float]:
    """Read the scale list from the model dir's own RIFE_HDv3.py.

    Each Practical-RIFE release ships its own scale_list (e.g. 4.25 uses
    [16, 8, 4, 2, 1] while 4.25.lite uses [32, 16, 8, 4, 1]).
    """
    import re

    rife_hdv3 = os.path.join(model_dir, "RIFE_HDv3.py")
    with open(rife_hdv3) as fobj:
        match = re.search(r"scale_list\s*=\s*\[([^\]]*)\]", fobj.read())
    if not match:
        raise RuntimeError("could not find scale_list in RIFE_HDv3.py")
    numerators = [float(num) for num in re.findall(r"\d+(?:\.\d+)?", match.group(1))]
    if not numerators:
        raise RuntimeError(f"could not parse scale_list in {rife_hdv3}")
    return numerators


def _warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward warping (from Practical-RIFE's model/warplayer.py, MIT).

    The grid is rebuilt on every call (no python-level caching) so that the
    traced graph keeps ``aten::size`` -> ``linspace`` dependencies and thus
    stays correct for dynamic input sizes.
    """
    horizontal = torch.linspace(
        -1.0, 1.0, img.shape[3]).view(1, 1, 1, img.shape[3]).expand(1, 1, img.shape[2], img.shape[3])
    vertical = torch.linspace(
        -1.0, 1.0, img.shape[2]).view(1, 1, img.shape[2], 1).expand(1, 1, img.shape[2], img.shape[3])
    grid = torch.cat([horizontal, vertical], 1)
    flow = torch.cat([
        flow[:, 0:1, :, :] / ((img.shape[3] - 1.0) / 2.0),
        flow[:, 1:2, :, :] / ((img.shape[2] - 1.0) / 2.0),
    ], 1)
    return torch.nn.functional.grid_sample(
        input=img,
        grid=(grid + flow).permute(0, 2, 3, 1),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


class _ExportableRIFE(nn.Module):
    """Trace-friendly rewrite of IFNet v4.x inference forward.

    Equivalent to IFNet.forward(..., fastmode=True, training=False); only the
    final merged frame is produced. Avoids the original's python list of
    (warped0, warped1) tuples which cannot be traced.
    """

    def __init__(self, ifnet: nn.Module, scale_list: list[float]) -> None:
        super().__init__()
        self.encode = ifnet.encode
        self.blocks = nn.ModuleList([ifnet.block0, ifnet.block1, ifnet.block2, ifnet.block3, ifnet.block4])
        self.scale_list = scale_list

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        img0 = x[:, :3]
        img1 = x[:, 3:]
        t = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])
        f0 = self.encode(img0)
        f1 = self.encode(img1)
        flow = None
        mask = None
        warped0 = img0
        warped1 = img1
        for i, block in enumerate(self.blocks):
            if flow is None:
                flow, mask, feat = block(torch.cat((img0, img1, f0, f1, t), 1), None, self.scale_list[i])
            else:
                wf0 = _warp(f0, flow[:, :2])
                wf1 = _warp(f1, flow[:, 2:4])
                fd, mask, feat = block(torch.cat((warped0, warped1, wf0, wf1, t, mask, feat), 1), flow, self.scale_list[i])
                flow = flow + fd
            warped0 = _warp(img0, flow[:, :2])
            warped1 = _warp(img1, flow[:, 2:4])
        mask = torch.sigmoid(mask)
        return warped0 * mask + warped1 * (1 - mask)


def load_rife(model_dir: str, scale_list: list[float], scale: float = 1.0) -> nn.Module:
    model_dir = os.path.abspath(model_dir)
    if not os.path.isfile(os.path.join(model_dir, "flownet.pkl")):
        raise RuntimeError(f"{model_dir} does not contain flownet.pkl")

    env_dir = _prepare_import_env(
        model_dir, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp")
    )
    sys.path.insert(0, os.path.abspath(model_dir))
    sys.path.insert(0, env_dir)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # RIFE_HDv3.py does its own cuda/cpu pick; force cpu for export.
    torch.cuda.is_available = lambda: False  # type: ignore[assignment]
    try:
        ifnet_mod = __import__("IFNet_HDv3")
        ifnet = ifnet_mod.IFNet()
        with open(os.path.join(model_dir, "flownet.pkl"), "rb") as fobj:
            state = torch.load(fobj, map_location="cpu")
        state = {k.replace("module.", ""): v for k, v in state.items()}
        ifnet.load_state_dict(state, strict=False)
        ifnet.eval()
        if scale_list is None:
            scale_list = _parse_scale_list(model_dir)
    finally:
        for path in (os.path.abspath(model_dir), env_dir, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
            if path in sys.path:
                sys.path.remove(path)
    for param in ifnet.parameters():
        param.requires_grad_(False)
    return _ExportableRIFE(ifnet, [s / scale for s in scale_list])


def export(
    model_dir: str,
    onnx_path: str,
    example_size: tuple[int, int] = (512, 384),
    scale: float = 1.0,
) -> str:
    wrapper = load_rife(model_dir, None, scale)

    height, width = example_size
    example = (
        torch.rand(1, 6, height, width, device="cpu"),
        torch.ones(1, 1, 1, 1) * 0.5,
    )
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example,
            os.path.abspath(onnx_path),
            input_names=["x", "timestep"],
            output_names=["frame"],
            dynamic_axes={"x": {2: "height", 3: "width"}},
            opset_version=17,
            do_constant_folding=True,
            # The new torch.export-based exporter (default since torch 2.9)
            # chokes on the symbolic shape guards inside the warp grid math;
            # the TorchScript tracer handles it fine.
            dynamo=False,
        )
    return os.path.abspath(onnx_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="directory with flownet.pkl / RIFE_HDv3.py (a Practical-RIFE release dir)")
    parser.add_argument("out", nargs="?", default=None, help="output onnx path")
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    out = args.out
    base = os.path.basename(os.path.normpath(args.model_dir)) or "rife"
    if out is None:
        out = f"rife-{base}-s{args.scale}.onnx"
    print(export(args.model_dir, out, scale=args.scale))


if __name__ == "__main__":
    main()
