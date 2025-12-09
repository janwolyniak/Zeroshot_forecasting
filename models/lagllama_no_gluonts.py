# lagllama_no_lightning.py
# Pure-PyTorch, GluonTS-free, Lightning-free zero-shot inference for Lag-Llama.

import math, sys, types
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

DEVICE = torch.device("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
DTYPE  = torch.float32



# --- Safe unpickling shim for PyTorch 2.6+ ---
import sys, types
def _install_gluonts_dummy_symbols():
    pkg = types.ModuleType("gluonts")
    torchmod = types.ModuleType("gluonts.torch")
    distmod  = types.ModuleType("gluonts.torch.distributions")
    studmod  = types.ModuleType("gluonts.torch.distributions.studentT")
    gaussmod = types.ModuleType("gluonts.torch.distributions.gaussian")
    nbmod    = types.ModuleType("gluonts.torch.distributions.neg_binomial")

    class StudentTOutput: pass
    class GaussianOutput: pass
    class NegativeBinomialOutput: pass

    studmod.StudentTOutput = StudentTOutput
    gaussmod.GaussianOutput = GaussianOutput
    nbmod.NegativeBinomialOutput = NegativeBinomialOutput

    sys.modules.update({
        "gluonts": pkg,
        "gluonts.torch": torchmod,
        "gluonts.torch.distributions": distmod,
        "gluonts.torch.distributions.studentT": studmod,
        "gluonts.torch.distributions.gaussian": gaussmod,
        "gluonts.torch.distributions.neg_binomial": nbmod,
    })

    # allow-list for torch.load (PyTorch 2.6+)
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals([StudentTOutput, GaussianOutput, NegativeBinomialOutput])
    except Exception:
        pass

# --- Safe-load shim: ckpt references GluonTS distribution classes in its pickled metadata.
def _install_gluonts_dummy_symbols():
    pkg = types.ModuleType("gluonts")
    torchmod = types.ModuleType("gluonts.torch")
    distmod  = types.ModuleType("gluonts.torch.distributions")
    studmod  = types.ModuleType("gluonts.torch.distributions.studentT")
    gaussmod = types.ModuleType("gluonts.torch.distributions.gaussian")
    nbmod    = types.ModuleType("gluonts.torch.distributions.neg_binomial")
    class StudentTOutput: pass
    class GaussianOutput: pass
    class NegativeBinomialOutput: pass
    studmod.StudentTOutput = StudentTOutput
    gaussmod.GaussianOutput = GaussianOutput
    nbmod.NegativeBinomialOutput = NegativeBinomialOutput
    sys.modules.update({
        "gluonts": pkg,
        "gluonts.torch": torchmod,
        "gluonts.torch.distributions": distmod,
        "gluonts.torch.distributions.studentT": studmod,
        "gluonts.torch.distributions.gaussian": gaussmod,
        "gluonts.torch.distributions.neg_binomial": nbmod,
    })
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals([StudentTOutput, GaussianOutput, NegativeBinomialOutput])
    except Exception:
        pass

def load_ckpt_dict():
    _install_gluonts_dummy_symbols()
    path = hf_hub_download("time-series-foundation-models/Lag-Llama", "lag-llama.ckpt")
    # weights_only=True is safer on newer Torch; allowlist shim above keeps old ckpts happy.
    return torch.load(path, map_location=DEVICE, weights_only=False)

# Try common locations for the pure torch core (no Lightning).
def resolve_core_class():
    tried = []
    for mod_path, cls_name in [
        ("lag_llama.model",          "LagLlama"),           # common
        ("lag_llama.model",          "LagLlamaModel"),
        ("lag_llama.core",           "LagLlama"),
        ("lag_llama.network",        "LagLlama"),
        ("lag_llama",                "LagLlama"),
    ]:
        try:
            m = __import__(mod_path, fromlist=[cls_name])
            if hasattr(m, cls_name):
                return getattr(m, cls_name)
            tried.append(f"{mod_path}.{cls_name} (no class)")
        except Exception as e:
            tried.append(f"{mod_path}.{cls_name} -> {type(e).__name__}")
    raise ImportError("Could not locate Lag-Llama core class. Tried:\n  - " + "\n  - ".join(tried))

def build_core_from_ckpt(ckpt: Dict):
    Core = resolve_core_class()
    hparams = ckpt.get("hyper_parameters", {})
    # Prefer the explicit model kwargs if present (documented in issues)
    model_kwargs = (hparams.get("model_kwargs") or hparams)
    core = Core(**model_kwargs)  # pure torch module
    # Extract only the sub-dict that maps to the core.
    sd = ckpt["state_dict"]
    # Keys might be like "core.xxx", "model.xxx", or already flat.
    core_state = {}
    for k, v in sd.items():
        if k.startswith("core."):
            core_state[k[len("core."):]] = v
        elif k.startswith("model."):
            core_state[k[len("model."):]] = v
        elif not any(k.startswith(p) for p in ("_forward_module.", "trainer.", "optimizer.")):
            # last resort: hope it's already flat core params
            core_state[k] = v
    missing, unexpected = core.load_state_dict(core_state, strict=False)
    if missing:
        print(f"[warn] missing keys in core: {len(missing)} (first 5): {missing[:5]}")
    if unexpected:
        print(f"[warn] unexpected keys in core: {len(unexpected)} (first 5): {unexpected[:5]}")
    core.eval().to(DEVICE)
    return core, model_kwargs

# --- Time features (sin/cos) per paper
def _phase(x, period): 
    ang = 2 * math.pi * (x % period) / period
    return np.sin(ang), np.cos(ang)

def make_time_features(idx: pd.DatetimeIndex) -> np.ndarray:
    sec   = idx.second
    minu  = idx.minute
    hour  = idx.hour
    dow   = idx.dayofweek
    dom   = idx.day
    doy   = idx.dayofyear
    week  = idx.isocalendar().week.to_numpy()
    month = idx.month
    quart = ((idx.month - 1) // 3) + 1
    feats = []
    for arr, period in [(sec,60),(minu,60),(hour,24),(dow,7),(dom,31),(doy,366),(week,53),(month,12),(quart,4)]:
        s,c = _phase(np.asarray(arr, dtype=np.int64), period)
        feats += [s,c]
    return np.stack(feats, axis=1).astype("float32")  # [T, 18]

# Heuristic lag sets (good zero-shot defaults)
DEFAULT_LAGS_MIN = [1,2,3,4,5,6,7,8,9,10,12,15,20,30,60,120,240,360,720,1440,2880,10080]
DEFAULT_LAGS_H   = [1,2,3,4,5,6,7,8,12,24,36,48,72,96,120,144,168,336,720]
DEFAULT_LAGS_D   = [1,2,3,4,5,6,7,14,21,28,30,60,90,180,365]

def pick_lags(freq: str, context_length: int):
    f = (freq or "").lower()
    if "min" in f or f in ("t","1t"): base = DEFAULT_LAGS_MIN
    elif "h" in f:                    base = DEFAULT_LAGS_H
    elif "d" in f:                    base = DEFAULT_LAGS_D
    else:                             base = list(range(1, max(8, context_length//4)))
    return np.unique(np.array(sorted(base + list(range(1, min(context_length, 32)))))).astype(np.int64)

@dataclass
class Scale: mean: float; std: float
def zscore(x: np.ndarray) -> Tuple[np.ndarray, Scale]:
    mu = float(np.nanmean(x)); sd = float(np.nanstd(x) + 1e-6)
    return (x - mu) / sd, Scale(mu, sd)

def build_inputs(y: pd.Series, prediction_length: int, context_length: int, freq: Optional[str]):
    assert isinstance(y.index, pd.DatetimeIndex)
    freq = freq or (y.index.freqstr if y.index.freq is not None else pd.infer_freq(y.index) or "1H")
    y = y.astype("float32")
    if y.isna().any(): y = y.ffill().bfill()

    lags_seq = pick_lags(freq, context_length)
    L_attn   = int(context_length)
    L_full   = int(max(max(lags_seq), L_attn))

    # Construct a full index that includes the lag window + future
    step = (y.index[1] - y.index[0])
    full_index = pd.date_range(
        start=y.index[0] - step * (L_full - 1),
        periods=len(y) + (L_full - 1) + prediction_length,
        freq=(y.index.freq or step)
    )
    y_full = pd.Series(np.nan, index=full_index, dtype="float32")
    y_full.loc[y.index] = y.values

    tfull = make_time_features(full_index)

    past_raw = y_full.iloc[: (L_full + len(y))].to_numpy()[-(L_full + L_attn):]
    observed = (~np.isnan(past_raw)).astype("float32")
    fill = np.nanmean(past_raw); x = past_raw.copy(); x[np.isnan(x)] = fill
    x, scale = zscore(x)

    tf_past   = tfull[: (L_full + len(y))][-(L_full + L_attn):, :]
    tf_future = tfull[(L_full + len(y)) : (L_full + len(y) + prediction_length), :]

    batch = {
        "past_target": torch.tensor(x, device=DEVICE, dtype=DTYPE).unsqueeze(0),
        "past_observed_values": torch.tensor(observed, device=DEVICE, dtype=DTYPE).unsqueeze(0),
        "past_time_feat": torch.tensor(tf_past, device=DEVICE, dtype=DTYPE).unsqueeze(0),
        "future_time_feat": torch.tensor(tf_future, device=DEVICE, dtype=DTYPE).unsqueeze(0),
        "lags_seq": torch.tensor(lags_seq, device=DEVICE, dtype=torch.int64).unsqueeze(0),
        "context_length": torch.tensor([L_attn], device=DEVICE, dtype=torch.int64),
        "prediction_length": torch.tensor([prediction_length], device=DEVICE, dtype=torch.int64),
    }
    fut_idx = full_index[-prediction_length:]
    return batch, scale, fut_idx

@torch.no_grad()
def forecast_no_lightning(y: pd.Series, prediction_length: int, context_length: int = 512,
                          freq: Optional[str] = None, num_samples: int = 200,
                          rope_scale: Optional[float] = 1.5):
    ckpt = load_ckpt_dict()
    core, model_kwargs = build_core_from_ckpt(ckpt)

    # Optional: RoPE scaling if the core exposes it (helps > pretrain ctx). Repo recommends this. 
    if rope_scale is not None and hasattr(core, "rope") and hasattr(core.rope, "scale"):
        try: core.rope.scale = rope_scale
        except Exception: pass  # model variant without explicit scale attr

    batch, scale, fut_idx = build_inputs(y, prediction_length, context_length, freq)

    # Autoregressive roll with optional KV cache (if the core provides one).
    xb = batch["past_target"]; mb = batch["past_observed_values"]
    p_tf = batch["past_time_feat"]; f_tf = batch["future_time_feat"]
    lags = batch["lags_seq"]; P = int(batch["prediction_length"][0])

    cache = core.init_kv_cache(xb.size(0)) if hasattr(core, "init_kv_cache") else None

    means, draws = [], []
    x_hist, m_hist = xb.clone(), mb.clone()

    for t in range(P):
        out = core(
            past_target=x_hist,
            past_observed_values=m_hist,
            past_time_feat=p_tf,
            future_time_feat=f_tf[:, : t+1, :],
            lags_seq=lags,
            kv_cache=cache
        )
        # Typical keys in outputs (Student-T head)
        if isinstance(out, dict):
            mean = out.get("mean") or out.get("loc")
            scale_t = out.get("scale") or out.get("sigma")
            df = out.get("df") or out.get("nu")
        else:
            try:
                mean, scale_t, df = out
            except Exception as e:
                raise RuntimeError(f"Unexpected forward() return: {type(out)}") from e

        mu_t = mean[:, -1]  # [B]
        if scale_t is not None and df is not None:
            td = torch.distributions.StudentT(df[:, -1].clamp_min(2.01))
            samp = (td.rsample((num_samples,)).transpose(0,1) * scale_t[:, -1] + mu_t)  # [B,S]
        else:
            # deterministic fallback if only mean is available
            samp = mu_t.repeat(num_samples, 1).T

        means.append(mu_t.detach().cpu())
        draws.append(samp.detach().cpu())

        # teacher-forced mean rollout
        x_hist = torch.cat([x_hist, mu_t.unsqueeze(-1)], dim=1)
        m_hist = torch.cat([m_hist, torch.ones_like(mu_t.unsqueeze(-1))], dim=1)

    mean = torch.stack(means, dim=1)[0].numpy()
    smp  = torch.stack(draws, dim=0)[:, 0, :].numpy()

    mean = mean * scale.std + scale.mean
    smp  = smp  * scale.std + scale.mean

    out = pd.DataFrame({"mean": mean}, index=fut_idx)
    q = np.quantile(smp, [0.1, 0.5, 0.9], axis=1).T.astype("float32")
    out["q10"], out["q50"], out["q90"] = q[:,0], q[:,1], q[:,2]
    return out