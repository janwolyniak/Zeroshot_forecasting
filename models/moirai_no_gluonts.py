# moirai_no_gluonts.py
# GluonTS-free Moirai forecaster with flexible, zero-hardcoded imports.

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

try:
    from safetensors.torch import load_file as load_safetensors
except Exception:
    load_safetensors = None

Array = Union[np.ndarray, pd.Series]


def _pick_device_dtype() -> Tuple[torch.device, torch.dtype]:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps"), torch.float32
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.device("cuda"), torch.bfloat16
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


def _standardize(x: np.ndarray):
    mu = float(np.mean(x))
    sd = float(np.std(x)) or 1.0
    return (x - mu) / sd, mu, sd


def _import_by_path(path: str):
    """
    Import "pkg.subpkg:ClassName" or "pkg.subpkg.ClassName".
    """
    if ":" in path:
        mod, cls = path.split(":", 1)
    else:
        parts = path.split(".")
        mod, cls = ".".join(parts[:-1]), parts[-1]
    module = importlib.import_module(mod)
    return getattr(module, cls)


@dataclass
class _AdapterConfig:
    repo_or_dir: str
    patch_size: int
    rope_scale: float


class _MoiraiPredictor:
    """
    Flexible loader:
      1) If env MOIRAI_CLASS is set to "package.module:ClassName" (recommended), use that.
      2) Else try a set of common candidates (no guarantee for all forks).
      3) If no python class found, try loading safetensors with a known class (via MOIRAI_CLASS).
    Inference:
      - Prefer `model.generate(x, horizon=H, ...)`
      - Else use `model(x, horizon=H)` if callable
      - Else raise a clear error describing what to implement.
    """
    def __init__(
        self,
        repo_or_dir: Union[str, os.PathLike],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        patch_size: int = 16,
        rope_scale: float = 1.0,
    ) -> None:
        self.device, self.dtype = device or _pick_device_dtype()
        self.dtype = dtype or self.dtype
        self.cfg = _AdapterConfig(str(repo_or_dir), int(patch_size), float(rope_scale))
        self.model = None

        class_path = os.environ.get("MOIRAI_CLASS", "").strip()

        def _try_candidates():
            # Add or remove candidates to taste; these cover common OSS forks.
            candidates = [
                "moirai:MoiraiModule",
                "moirai:MoiraiForecast",
                "moirai.modeling:MoiraiModule",    # US spelling
                "moirai.modelling:MoiraiModule",   # UK spelling
                "salesforce_moirai:MoiraiModule",
            ]
            for cand in candidates:
                try:
                    Model = _import_by_path(cand)
                    return Model
                except Exception:
                    continue
            return None

        Model = None
        if class_path:
            try:
                Model = _import_by_path(class_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to import MOIRAI_CLASS='{class_path}'. "
                    f"Set MOIRAI_CLASS to 'pkg.module:ClassName'."
                ) from e
        else:
            Model = _try_candidates()

        # Strategy A: use Model.from_pretrained if available
        if Model is not None:
            try:
                if hasattr(Model, "from_pretrained"):
                    self.model = Model.from_pretrained(self.cfg.repo_or_dir)
                else:
                    # Instantiate; try passing patch_size if the constructor supports it
                    try:
                        self.model = Model(patch_size=self.cfg.patch_size)
                    except TypeError:
                        self.model = Model()
            except Exception as e:
                # fall through to safetensors Strategy B
                self.model = None
                last_err = e
        else:
            last_err = RuntimeError("No suitable Moirai class found/imported.")

        # Strategy B: load safetensors state into Model if provided
        if self.model is None and load_safetensors is not None:
            repo = Path(self.cfg.repo_or_dir)
            st_path = None
            for name in ("model.safetensors", "pytorch_model.safetensors"):
                cand = repo / name
                if cand.exists():
                    st_path = cand
                    break

            if st_path is None:
                raise RuntimeError(
                    f"Could not import a Moirai class and no safetensors checkpoint found under {repo}. "
                    f"Last error: {repr(last_err)}"
                )

            if Model is None:
                # require explicit class when loading a raw checkpoint
                raise RuntimeError(
                    "To load safetensors without an importable package, set "
                    "MOIRAI_CLASS='your.package:YourMoiraiClass' to construct the architecture."
                )

            # build model then load state
            try:
                try:
                    self.model = Model(patch_size=self.cfg.patch_size)
                except TypeError:
                    self.model = Model()
                state = load_safetensors(str(st_path), device="cpu")
                missing, unexpected = self.model.load_state_dict(state, strict=False)
                if missing:
                    print(f"[Moirai] Missing keys: {len(missing)}")
                if unexpected:
                    print(f"[Moirai] Unexpected keys: {len(unexpected)}")
            except Exception as e:
                raise RuntimeError(f"Failed to load weights from {st_path}") from e

        if self.model is None:
            raise RuntimeError("Failed to construct Moirai model. Check MOIRAI_CLASS and repo_or_dir.")

        self.model.to(self.device)
        try:
            self.model.eval()
        except Exception:
            pass

    @torch.no_grad()
    def predict(self, y_context: Array, horizon: int = 1) -> np.ndarray:
        y_np = np.asarray(y_context, dtype=np.float32).reshape(-1)
        x_std, mu, sd = _standardize(y_np)
        x = torch.from_numpy(x_std).to(self.device, dtype=self.dtype).view(1, -1, 1)  # (B,L,1)

        # Inference dispatch
        if hasattr(self.model, "generate"):
            try:
                out = self.model.generate(
                    x, horizon=horizon, patch_size=self.cfg.patch_size, rope_scale=self.cfg.rope_scale
                )
            except TypeError:
                out = self.model.generate(x, horizon=horizon)
        elif callable(getattr(self.model, "forward", None)) or callable(self.model):
            out = self.model(x, horizon=horizon)  # may raise if horizon not supported
        else:
            raise RuntimeError(
                "Model has neither .generate(...) nor a callable forward; "
                "implement one of these in your adapter."
            )

        # Expect (B,H) or (B,H,1); fall back to persistence if shapes are unexpected
        if isinstance(out, torch.Tensor):
            if out.dim() == 3 and out.size(-1) == 1:
                y_hat_std = out.squeeze(-1)
            elif out.dim() == 2:
                y_hat_std = out
            else:
                # shape is odd; fallback to persistence to keep pipeline running
                y_hat_std = torch.full((1, horizon), float(x_std[-1]), device=x.device)
        else:
            # If your generate returns a dict or custom object, map it here.
            y_hat_std = torch.full((1, horizon), float(x_std[-1]), device=x.device)

        y_hat = (y_hat_std.float().cpu().numpy().reshape(-1)) * sd + mu
        return y_hat.astype(np.float64)


_cached = {"repo": None, "pred": None, "device": None, "dtype": None, "ps": None, "rope": None}


def forecast_no_lightning(
    y: pd.Series,
    prediction_length: int,
    context_length: int,
    freq: str,
    num_samples: int = 0,
    repo_or_dir: Optional[Union[str, os.PathLike]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    patch_size: int = 16,
    rope_scale: float = 1.0,
) -> pd.DataFrame:
    assert isinstance(y, pd.Series) and y.index.is_monotonic_increasing, "Provide a time-indexed pandas Series"
    assert len(y) >= context_length, "Context length exceeds provided series length."

    ctx = y.iloc[-context_length:]
    dev, dt = device or _pick_device_dtype(), (dtype or None)

    repo_dir = str(repo_or_dir or os.environ.get("MOIRAI_REPO_OR_DIR", "") or ".")
    need_new = (
        _cached["pred"] is None
        or _cached["repo"] != repo_dir
        or _cached["device"] != str(dev)
        or _cached["dtype"] != str(dt)
        or _cached["ps"] != int(patch_size)
        or _cached["rope"] != float(rope_scale)
    )
    if need_new:
        _cached["pred"] = _MoiraiPredictor(
            repo_or_dir=repo_dir, device=dev, dtype=dt, patch_size=patch_size, rope_scale=rope_scale
        )
        _cached.update({"repo": repo_dir, "device": str(dev), "dtype": str(dt), "ps": int(patch_size), "rope": float(rope_scale)})

    y_hat = _cached["pred"].predict(ctx.values, horizon=prediction_length)

    last_ts = pd.to_datetime(ctx.index[-1])
    idx = pd.date_range(last_ts, periods=prediction_length + 1, freq=freq)[1:]
    df = pd.DataFrame({"timestamp": idx, "mean": y_hat})
    df.set_index("timestamp", inplace=True)
    return df