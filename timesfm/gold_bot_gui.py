"""
TimesFM Gold Bot — XAUUSD Rise/Fall trader for Deriv
=====================================================
Same engine/GUI architecture as the TimesFM Deriv digit bot, rebuilt for Gold.

Combines five setups into one ensemble decision:
  S1  Quantile-band breakout        (TimesFM p10/p50/p90 as a dynamic envelope)
  S2  Multivariate macro regime     (Gold + DXY-proxy + Silver, joint forecast)
  S3  Event covariates + blackout   (NFP / CPI / FOMC as past-and-future covariates)
  S4  Multi-timeframe confluence    (H1 + M15 + M5 forecasts must agree)
  S5  Quantile-based risk sizing    (stake shrinks as the forecast band widens)

Forecaster fallback chain: TimesFM 3.0  ->  TimesFM 2.5  ->  statistical model.

Includes a walk-forward BACKTEST that uses exactly the same decision code as
live trading, so you can measure the edge before risking money.

Requirements:  pip install websockets numpy
Optional:      pip install -e ./timesfm-3.0.0[torch]     (TimesFM 3.0 + 2.5)
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import asyncio
import threading
import json
import time
import os
import sys
import queue
import math
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple
from collections import deque
from urllib.request import Request, urlopen
from urllib.error import HTTPError

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False


# ─────────────────────────────────────────────
#  THEME
# ─────────────────────────────────────────────
DARK  = "#0D0F14"
PANEL = "#13161E"
CARD  = "#1A1E2A"
BORD  = "#252A38"
ACCENT= "#00D4FF"
GOLD  = "#FFC940"
GREEN = "#00FF88"
RED   = "#FF4466"
WARN  = "#FFB020"
TEXT  = "#E8EAED"
MUTED = "#6B7385"
FONT_MONO = ("Consolas", 10)
FONT_SM   = ("Segoe UI", 9)


# ─────────────────────────────────────────────
#  MARKET CONSTANTS
# ─────────────────────────────────────────────
GOLD_SYMBOL   = "frxXAUUSD"
SILVER_SYMBOL = "frxXAGUSD"
# ICE US Dollar Index formula — rebuilt from Deriv FX pairs (DXY itself is not on Deriv)
DXY_WEIGHTS = {
    "frxEURUSD": -0.576,
    "frxUSDJPY":  0.136,
    "frxGBPUSD": -0.119,
    "frxUSDCAD":  0.091,
    "frxUSDSEK":  0.042,
    "frxUSDCHF":  0.036,
}
TF_SEC = {"M5": 300, "M15": 900, "H1": 3600}
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
Z_QUANT = [-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816]


def utc_now() -> float:
    return time.time()


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class GoldBotConfig:
    # Connection
    api_token: str = ""
    app_id: str = "1089"
    account_id: str = ""
    use_real_account: bool = False

    # Market / contract
    symbol: str = GOLD_SYMBOL
    contract_mode: str = "RISE_FALL"      # RISE_FALL | MULTIPLIER
    duration_min: int = 15                # Rise/Fall expiry in minutes
    multiplier: int = 50                  # Multiplier mode only
    max_hold_min: int = 60                # Multiplier mode: force-close after this
    min_payout_ratio: float = 0.70        # skip Rise/Fall if payout < 70% of stake

    # Forecaster
    use_timesfm: bool = True
    model_version: str = "auto"           # auto | 3.0 | 2.5 | stats
    device: str = "auto"                  # auto | cpu | cuda | mps
    ctx_m5: int = 512
    ctx_m15: int = 256
    ctx_h1: int = 256
    horizon_bars: int = 12

    # Setups
    use_s1_band: bool = True
    use_s2_macro: bool = True
    use_s3_events: bool = True
    use_s4_mtf: bool = True
    use_s5_sizing: bool = True
    w_prob: float = 1.0
    w_band: float = 0.8
    w_macro: float = 0.6
    w_mtf: float = 1.0
    require_mtf_agreement: bool = True
    min_confidence: float = 62.0
    min_prob: float = 0.56
    event_pre_min: int = 30
    event_post_min: int = 45
    events_text: str = (
        "# One event per line, UTC:  YYYY-MM-DD HH:MM  LABEL\n"
        "# Check a real calendar (e.g. forexfactory.com) and edit these.\n"
        "2026-10-02 12:30 NFP\n"
    )
    session_start_utc: int = 7            # London open
    session_end_utc: int = 20             # NY afternoon

    # Risk
    risk_pct: float = 1.0                 # % of balance per trade (S5)
    base_stake: float = 5.0               # used when S5 sizing is off
    min_stake: float = 1.0
    max_stake: float = 100.0
    use_martingale: bool = False
    martingale_multiplier: float = 2.0
    max_martingale_steps: int = 2
    max_daily_loss: float = 150.0
    max_daily_profit: float = 500.0
    max_consecutive_losses: int = 4
    max_trades_per_day: int = 30
    cooldown_bars: int = 1
    max_retries: int = 2

    # Backtest
    bt_bars: int = 3000
    bt_step: int = 2
    bt_payout: float = 0.85
    bt_start_balance: float = 1000.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GoldBotConfig":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GoldBotConfig":
        if not os.path.exists(path):
            return cls()
        try:
            with open(path) as f:
                return cls.from_dict(json.load(f))
        except Exception:
            return cls()


# ─────────────────────────────────────────────
#  DATA HELPERS
# ─────────────────────────────────────────────
@dataclass
class Candles:
    epoch: "np.ndarray"
    open: "np.ndarray"
    high: "np.ndarray"
    low: "np.ndarray"
    close: "np.ndarray"

    @classmethod
    def from_api(cls, rows: list) -> "Candles":
        rows = sorted(rows, key=lambda r: int(r["epoch"]))
        return cls(
            epoch=np.array([int(r["epoch"]) for r in rows], dtype=np.int64),
            open=np.array([float(r["open"]) for r in rows]),
            high=np.array([float(r["high"]) for r in rows]),
            low=np.array([float(r["low"]) for r in rows]),
            close=np.array([float(r["close"]) for r in rows]),
        )

    def __len__(self):
        return len(self.epoch)

    def closed_until(self, now_epoch: float, gran: int) -> "Candles":
        """Only candles fully closed at now_epoch (no look-ahead)."""
        n = int(np.searchsorted(self.epoch + gran, now_epoch, side="right"))
        return self.slice(0, n)

    def slice(self, a: int, b: int) -> "Candles":
        return Candles(self.epoch[a:b], self.open[a:b], self.high[a:b],
                       self.low[a:b], self.close[a:b])


def align_ffill(target_epochs: "np.ndarray", src: Optional[Candles]) -> Optional["np.ndarray"]:
    """Align src closes to target epochs with forward fill. NaN where no data yet."""
    if src is None or len(src) == 0:
        return None
    idx = np.searchsorted(src.epoch, target_epochs, side="right") - 1
    out = np.full(len(target_epochs), np.nan)
    ok = idx >= 0
    out[ok] = src.close[idx[ok]]
    return out


def build_dxy_proxy(target_epochs, fx: Dict[str, Candles]) -> Optional["np.ndarray"]:
    """Geometric DXY replica from whatever DXY component pairs are available."""
    log_sum = np.zeros(len(target_epochs))
    used_w = 0.0
    for sym, w in DXY_WEIGHTS.items():
        series = align_ffill(target_epochs, fx.get(sym))
        if series is None or np.isnan(series).mean() > 0.2:
            continue
        series = _ffill_nan(series)
        log_sum += w * np.log(series)
        used_w += abs(w)
    if used_w < 0.5:          # need at least EURUSD + something
        return None
    total_w = sum(abs(w) for w in DXY_WEIGHTS.values())
    return 50.14348112 * np.exp(log_sum * (total_w / used_w))


def _ffill_nan(a: "np.ndarray") -> "np.ndarray":
    a = a.copy()
    if np.isnan(a).all():
        return a
    first = np.argmax(~np.isnan(a))
    a[:first] = a[first]
    for i in range(1, len(a)):
        if np.isnan(a[i]):
            a[i] = a[i - 1]
    return a


# ─────────────────────────────────────────────
#  ECONOMIC EVENTS (S3)
# ─────────────────────────────────────────────
def parse_events(text: str) -> List[Tuple[float, str]]:
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            dt = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y-%m-%d %H:%M")
            dt = dt.replace(tzinfo=timezone.utc)
            events.append((dt.timestamp(), " ".join(parts[2:]) or "EVENT"))
        except ValueError:
            continue
    return sorted(events)


def event_blackout(now_epoch: float, events, pre_min: int, post_min: int) -> Optional[str]:
    for ev_t, label in events:
        if ev_t - pre_min * 60 <= now_epoch <= ev_t + post_min * 60:
            mins = (ev_t - now_epoch) / 60
            when = f"in {mins:.0f}m" if mins > 0 else f"{-mins:.0f}m ago"
            return f"{label} ({when})"
    return None


def event_covariates(epochs: "np.ndarray", events) -> "np.ndarray":
    """2 known-future channels: proximity bump (±30m) and 2-hour pre-event ramp."""
    prox = np.zeros(len(epochs))
    ramp = np.zeros(len(epochs))
    for ev_t, _ in events:
        d = epochs - ev_t
        prox = np.maximum(prox, np.exp(-0.5 * (d / 1800.0) ** 2))
        pre = (d < 0) & (d > -7200)
        ramp[pre] = np.maximum(ramp[pre], 1.0 + d[pre] / 7200.0)
    return np.vstack([prox, ramp]).astype(np.float32)


# ─────────────────────────────────────────────
#  FORECASTER  (TimesFM 3.0 -> 2.5 -> statistical)
# ─────────────────────────────────────────────
# Loaded models survive engine restarts: (model_version, device) -> (kind, model)
_MODEL_CACHE: Dict[tuple, tuple] = {}


class Forecaster:
    """
    Unified interface:
        forecast(contexts=[(V,T) arrays], horizon, pf_covs=[(C,T+H)] | None)
        -> list of (median (V,H), quantiles (V,H,9))
    """

    def __init__(self, cfg: GoldBotConfig, log=print):
        self.cfg = cfg
        self.log = log
        self.kind = "stats"
        self.model = None
        self._lock = threading.Lock()

    def load(self):
        if not self.cfg.use_timesfm or self.cfg.model_version == "stats":
            self.log("Forecaster: statistical model (TimesFM disabled)", "warn")
            return
        if self.model is not None:
            return
        key = (self.cfg.model_version, self.cfg.device)
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            self.kind, self.model = cached
            self.log(f"✅ TimesFM {self.kind} reused from memory", "win")
            return
        order = ["3.0", "2.5"] if self.cfg.model_version == "auto" else [self.cfg.model_version]
        for v in order:
            try:
                if v == "3.0":
                    self._load_v3()
                else:
                    self._load_v25()
                _MODEL_CACHE[key] = (self.kind, self.model)
                return
            except Exception as e:
                self.log(f"TimesFM {v} unavailable: {e}", "warn")
        self.log("Falling back to statistical forecaster", "warn")

    def _device(self):
        return None if self.cfg.device == "auto" else self.cfg.device

    def _load_v3(self):
        import torch  # noqa
        from timesfm3 import TimesFM3Forecaster
        self.log("Loading TimesFM 3.0 (google/timesfm-3.0-pytorch)… first run downloads weights")
        self.model = TimesFM3Forecaster.from_pretrained(
            "google/timesfm-3.0-pytorch", device=self._device(), per_core_batch_size=8)
        self.kind = "3.0"
        self.log("✅ TimesFM 3.0 ready — multivariate + covariates active", "win")

    def _load_v25(self):
        import torch
        import timesfm
        self.log("Loading TimesFM 2.5 (200M)…")
        torch.set_float32_matmul_precision("high")
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        m.compile(timesfm.ForecastConfig(
            max_context=1024, max_horizon=256, normalize_inputs=True,
            use_continuous_quantile_head=True, force_flip_invariance=True,
            infer_is_positive=True, fix_quantile_crossing=True))
        self.model = m
        self.kind = "2.5"
        self.log("✅ TimesFM 2.5 ready (univariate per series; covariates ignored)", "win")

    @property
    def supports_covariates(self) -> bool:
        return self.kind == "3.0"

    def forecast(self, contexts, horizon: int, pf_covs=None):
        contexts = [np.atleast_2d(np.asarray(c, dtype=np.float32)) for c in contexts]
        with self._lock:
            if self.kind == "3.0":
                try:
                    return self._fc_v3(contexts, horizon, pf_covs)
                except Exception as e:
                    self.log(f"TimesFM 3.0 forecast error ({e}) — stats fallback this bar", "warn")
            elif self.kind == "2.5":
                try:
                    return self._fc_v25(contexts, horizon)
                except Exception as e:
                    self.log(f"TimesFM 2.5 forecast error ({e}) — stats fallback this bar", "warn")
        return [self._fc_stats(c, horizon) for c in contexts]

    def _fc_v3(self, contexts, horizon, pf_covs):
        kw = dict(contexts=contexts, horizon=horizon, return_quantiles=True,
                  use_symmetric_averaging=False, make_positive=False)
        if pf_covs is not None:
            kw.update(past_future_covariates=pf_covs, padding_mode="edge")
        out = []
        for o in self.model.predict_batch(**kw):
            med = np.atleast_2d(o.forecast)
            q = o.quantiles if o.quantiles.ndim == 3 else o.quantiles[None]
            out.append((med, np.sort(q, axis=-1)))
        return out

    def _fc_v25(self, contexts, horizon):
        flat = [row for c in contexts for row in c]
        point, quant = self.model.forecast(horizon=horizon, inputs=flat)
        quant = np.sort(np.asarray(quant)[:, :, 1:10], axis=-1)
        point = np.asarray(point)
        out, i = [], 0
        for c in contexts:
            v = c.shape[0]
            out.append((point[i:i + v], quant[i:i + v]))
            i += v
        return out

    @staticmethod
    def _fc_stats(ctx, horizon):
        """Damped-drift lognormal quantiles — honest baseline when no model."""
        meds, qs = [], []
        h = np.arange(1, horizon + 1)
        z = np.array(Z_QUANT)
        for row in ctx:
            x = row[~np.isnan(row)][-300:]
            last = float(x[-1])
            lr = np.diff(np.log(np.maximum(x, 1e-9)))
            sig = float(np.std(lr[-100:])) or 1e-6
            mu = float(np.mean(lr[-20:])) * 0.5
            cum_mu = mu * np.minimum(h, 6) + 0 * h
            s = sig * np.sqrt(h)
            q = last * np.exp(cum_mu[:, None] + s[:, None] * z[None, :])
            meds.append(q[:, 4])
            qs.append(q)
        return np.array(meds), np.array(qs)


def prob_above(level: float, qvals: "np.ndarray") -> float:
    """P(future > level) from the 9 forecast quantiles (piecewise-linear CDF)."""
    q = np.asarray(qvals, dtype=float)
    lo_gap = max(q[1] - q[0], 1e-9)
    hi_gap = max(q[-1] - q[-2], 1e-9)
    if level <= q[0]:
        F = max(0.02, 0.1 - 0.1 * (q[0] - level) / lo_gap)
    elif level >= q[-1]:
        F = min(0.98, 0.9 + 0.1 * (level - q[-1]) / hi_gap)
    else:
        F = float(np.interp(level, q, QUANTILES))
    return 1.0 - F


# ─────────────────────────────────────────────
#  STRATEGY (pure — shared by live trading and backtest)
# ─────────────────────────────────────────────
@dataclass
class Decision:
    direction: str = "FLAT"            # RISE | FALL | FLAT
    confidence: float = 0.0
    prob_up: float = 0.5
    score: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    price: float = 0.0
    regime: str = "—"
    size_factor: float = 1.0
    sl_dist: float = 0.0
    tp_dist: float = 0.0
    chart: Optional[dict] = None


class GoldStrategy:
    def __init__(self, cfg: GoldBotConfig, fc: Forecaster):
        self.cfg = cfg
        self.fc = fc
        self.width_hist: deque = deque(maxlen=300)

    @property
    def dur_bars(self) -> int:
        return max(1, math.ceil(self.cfg.duration_min / 5))

    def evaluate(self, now_epoch: float, m5: Candles, m15: Optional[Candles],
                 h1: Optional[Candles], aux: Dict[str, Candles], events,
                 gate: bool = True) -> Decision:
        cfg = self.cfg
        d = Decision()
        if m5 is None or len(m5) < 64:
            d.reasons.append("not enough M5 history")
            return d
        price = float(m5.close[-1])
        d.price = price

        # ── Gates: session + S3 event blackout ──
        hour = datetime.fromtimestamp(now_epoch, tz=timezone.utc).hour
        wd = datetime.fromtimestamp(now_epoch, tz=timezone.utc).weekday()
        in_session = cfg.session_start_utc <= hour < cfg.session_end_utc and wd < 5
        blackout = (event_blackout(now_epoch, events, cfg.event_pre_min, cfg.event_post_min)
                    if cfg.use_s3_events else None)

        # ── Build M5 context (multivariate if S2) ──
        T = min(cfg.ctx_m5, len(m5))
        sl = m5.slice(len(m5) - T, len(m5))
        rows, names = [sl.close], ["gold"]
        if cfg.use_s2_macro:
            dxy = build_dxy_proxy(sl.epoch, aux)
            if dxy is not None:
                rows.append(dxy); names.append("dxy")
            ag = align_ffill(sl.epoch, aux.get(SILVER_SYMBOL))
            if ag is not None and np.isnan(ag).mean() < 0.2:
                rows.append(_ffill_nan(ag)); names.append("silver")
        ctx = np.vstack(rows).astype(np.float32)

        H = max(cfg.horizon_bars, self.dur_bars)
        k = self.dur_bars - 1

        pf_full = None
        if cfg.use_s3_events and events and self.fc.supports_covariates:
            fut = sl.epoch[-1] + 300 * np.arange(1, H + 1)
            pf_full = event_covariates(np.concatenate([sl.epoch, fut]), events)

        contexts = [ctx]
        pfs = [pf_full] if pf_full is not None else None
        if cfg.use_s1_band:
            contexts.append(ctx[:, :-1])            # forecast as of previous bar
            if pfs is not None:
                pfs.append(pf_full[:, :T - 1 + H])
        res = self.fc.forecast(contexts, H, pfs)
        med, q = res[0]
        qg = q[0]                                   # gold quantiles (H, 9)
        qk = qg[k]

        # ── Core probability from the quantile fan ──
        prob_up = prob_above(price, qk)
        d.prob_up = prob_up
        s_prob = clamp((prob_up - 0.5) / 0.25)

        # ── Regime from band width ──
        width = max((qk[8] - qk[0]) / price, 1e-9)
        self.width_hist.append(width)
        med_w = float(np.median(self.width_hist)) if len(self.width_hist) >= 20 else width
        ratio = width / med_w
        d.regime = "TIGHT" if ratio < 0.8 else ("WIDE" if ratio > 1.2 else "NORMAL")

        comps = {"Prob": s_prob}
        weights = {"Prob": cfg.w_prob}

        # ── S1: quantile-band breakout vs previous bar's forecast ──
        pos_prev = None
        if cfg.use_s1_band and len(res) > 1:
            qp = res[1][1][0][0]                    # prev forecast, gold, step 0, (9,)
            p10, p50, p90 = qp[0], qp[4], qp[8]
            half = max((p90 - p10) / 2, 1e-9)
            pos_prev = (price - p10) / max(p90 - p10, 1e-9)
            if price > p90:
                brk = 1.0
            elif price < p10:
                brk = -1.0
            else:
                brk = clamp((price - p50) / half) * 0.4
            if d.regime == "TIGHT":
                s1 = -brk                            # low vol: fade the poke
            elif d.regime == "WIDE":
                s1 = brk                             # high vol: follow it
            else:
                s1 = brk * 0.6
            comps["S1 Band"] = s1
            weights["S1 Band"] = cfg.w_band
            if abs(brk) == 1.0:
                d.reasons.append(f"S1 {'upper' if brk > 0 else 'lower'} band break ({d.regime} → "
                                 f"{'follow' if s1 * brk > 0 else 'fade'})")

        # ── S2: multivariate macro regime ──
        if cfg.use_s2_macro and len(names) > 1:
            parts, pw = 0.0, 0.0
            for name, w, sign in (("dxy", 0.6, -1.0), ("silver", 0.4, 1.0)):
                if name in names:
                    i = names.index(name)
                    last = float(ctx[i, -1])
                    qi = q[i][k]
                    half = max((qi[8] - qi[0]) / 2, 1e-12)
                    z = clamp((float(med[i][k]) - last) / half * 2)
                    parts += w * sign * z
                    pw += w
            if pw:
                s2 = parts / pw
                comps["S2 Macro"] = s2
                weights["S2 Macro"] = cfg.w_macro
                gold_dir = np.sign(float(med[0][k]) - price)
                if gold_dir != 0 and np.sign(s2) == -gold_dir and abs(s2) > 0.3:
                    d.reasons.append("S2 divergence: gold path disagrees with DXY/silver")

        # ── S4: multi-timeframe confluence ──
        mtf_dirs = []
        if cfg.use_s4_mtf and m15 is not None and h1 is not None and len(m15) >= 48 and len(h1) >= 48:
            c15 = m15.close[-cfg.ctx_m15:]
            c1h = h1.close[-cfg.ctx_h1:]
            r = self.fc.forecast([c15[None, :], c1h[None, :]], 4)
            zs = []
            for (mm, qq), last in zip(r, (c15[-1], c1h[-1])):
                half = max((qq[0][-1][8] - qq[0][-1][0]) / 2, 1e-9)
                z = clamp((float(mm[0][-1]) - float(last)) / half * 2)
                zs.append(z)
                mtf_dirs.append(np.sign(z))
            s4 = float(np.mean(zs))
            comps["S4 MTF"] = s4
            weights["S4 MTF"] = cfg.w_mtf

        # ── Ensemble ──
        tw = sum(weights.values())
        score = sum(comps[n] * weights[n] for n in comps) / tw if tw else 0.0
        d.score = score
        d.components = comps
        direction = "RISE" if score > 0 else "FALL"
        d.confidence = 50 + 50 * abs(score)
        dir_sign = 1 if direction == "RISE" else -1
        dir_prob = prob_up if direction == "RISE" else 1 - prob_up

        # ── S5: sizing + stop/target distances from quantiles ──
        d.size_factor = clamp(med_w / width, 0.5, 1.5) if cfg.use_s5_sizing else 1.0
        if direction == "RISE":
            d.sl_dist, d.tp_dist = max(price - qk[0], 0), max(qk[8] - price, 0)
        else:
            d.sl_dist, d.tp_dist = max(qk[8] - price, 0), max(price - qk[0], 0)

        tail = m5.close[-60:]
        d.chart = {"hist": tail.tolist(), "p10": qg[:, 0].tolist(),
                   "p50": qg[:, 4].tolist(), "p90": qg[:, 8].tolist(), "k": k}

        # ── Final gating ──
        veto = []
        if not in_session:
            veto.append(f"outside session {cfg.session_start_utc:02d}–{cfg.session_end_utc:02d} UTC")
        if blackout:
            veto.append(f"S3 event blackout: {blackout}")
        if d.confidence < cfg.min_confidence:
            veto.append(f"confidence {d.confidence:.1f} < {cfg.min_confidence}")
        if dir_prob < cfg.min_prob:
            veto.append(f"P({direction.lower()}) {dir_prob:.2f} < {cfg.min_prob}")
        if cfg.use_s4_mtf and cfg.require_mtf_agreement and mtf_dirs:
            if not all(s == dir_sign for s in mtf_dirs):
                veto.append("S4 timeframes disagree")
            if pos_prev is not None and ((dir_sign > 0 and pos_prev < 0.5) or (dir_sign < 0 and pos_prev > 0.5)):
                veto.append("S4 M5 not in the right half of its band")
        if "S2 Macro" in comps and comps["S2 Macro"] * dir_sign < -0.5:
            veto.append("S2 macro strongly against")

        if veto and gate:
            d.reasons = veto + d.reasons
            d.direction = "FLAT"
        else:
            d.direction = direction
        return d


# ─────────────────────────────────────────────
#  TRADING ENGINE (async, runs in a thread)
# ─────────────────────────────────────────────
class TradingEngine:
    def __init__(self, config: GoldBotConfig, msg_queue: queue.Queue, mode: str = "live"):
        self.config = config
        self.q = msg_queue
        self.mode = mode                        # live | backtest
        self._stop = threading.Event()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.ws = None
        self._pending: Dict[int, asyncio.Future] = {}
        self._req_id = 0
        self._cooldown_until_bar = -1
        self._orphan: Optional[dict] = None    # contract left open by a dropped connection
        self.currency = "USD"

        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.daily_pnl = 0.0
        self.day_trades = 0
        self.day_key = None
        self.balance = 0.0
        self.recovery_step = 0
        self.consecutive_losses = 0

        self.events = parse_events(config.events_text)
        self.forecaster = Forecaster(config, self._log)
        self.strategy = GoldStrategy(config, self.forecaster)
        self._bad_symbols = set()

    # ── public ──
    def stop(self):
        self._stop.set()

    # ── messaging ──
    def _emit(self, kind: str, **kw):
        self.q.put({"kind": kind, **kw})

    def _log(self, msg: str, level: str = "info"):
        self._emit("log", msg=f"[{ts()}] {msg}", level=level)
        print(f"[{level.upper()}] {msg}")

    # ── OTP session (same flow as the digit bot) ──
    def _get_otp_url(self) -> str:
        otp_url = (f"https://api.derivws.com/trading/v1/options/accounts"
                   f"/{self.config.account_id}/otp")
        headers = {"Deriv-App-ID": self.config.app_id,
                   "Authorization": f"Bearer {self.config.api_token}"}
        req = Request(otp_url, data=None, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            raise RuntimeError(f"OTP request failed HTTP {e.code}: {e.read().decode()}") from e
        url = self._find_wss_url(body)
        if url is None:
            raise RuntimeError(f"OTP response contained no wss:// URL: {body}")
        return url

    @staticmethod
    def _find_wss_url(value) -> Optional[str]:
        if isinstance(value, str):
            return value if value.startswith("wss://") else None
        if isinstance(value, dict):
            value = list(value.values())
        if isinstance(value, list):
            for v in value:
                found = TradingEngine._find_wss_url(v)
                if found:
                    return found
        return None

    def run_in_thread(self):
        if not self.config.account_id:
            self._emit("stopped", reason="Account ID is required for the OTP connection.")
            return
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        attempt = 0
        try:
            while not self._stop.is_set():
                try:
                    self._log("Fetching OTP session URL…")
                    ws_url = self._get_otp_url()   # OTP URLs are single-use: fetch fresh each time
                except Exception as e:
                    if self.mode != "live" or attempt >= self.MAX_RECONNECTS:
                        self._emit("stopped", reason=f"OTP fetch failed: {e}")
                        return
                    ws_url = None
                    self._log(f"OTP fetch failed: {e}", "warn")
                if ws_url is not None:
                    started = time.monotonic()
                    if self.loop.run_until_complete(self._run(ws_url)) != "dropped":
                        return
                    if time.monotonic() - started > 120:   # it was a healthy session
                        attempt = 0
                if self.mode != "live" or self._stop.is_set():
                    return
                attempt += 1
                if attempt > self.MAX_RECONNECTS:
                    self._log(f"Gave up after {self.MAX_RECONNECTS} reconnect attempts.", "error")
                    self._emit("stopped", reason="Feed lost — reconnect attempts exhausted")
                    return
                delay = min(5 * 2 ** (attempt - 1), 60)
                self._log(f"🔄 Reconnecting in {delay}s (attempt {attempt}/{self.MAX_RECONNECTS})…", "warn")
                self._emit("status", text="RECONNECTING", color=WARN)
                if self._stop.wait(delay):
                    return
        finally:
            self.loop.close()
            self._emit("stopped", reason="Engine loop exited")

    MAX_RECONNECTS = 10

    async def _run(self, ws_url: str) -> str:
        """Returns 'dropped' if the live feed died (caller reconnects), else 'done'."""
        self._log("Connecting…")
        outcome = "done"
        try:
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=30,
                                          close_timeout=5, max_size=2**24) as ws:
                self.ws = ws
                self._tick_queue: asyncio.Queue = asyncio.Queue()
                dispatch = asyncio.ensure_future(self._dispatcher())
                await asyncio.sleep(0)
                await self._authenticate()
                self._emit("status", text="LOADING MODEL", color=WARN)
                await self.loop.run_in_executor(None, self.forecaster.load)
                self._emit("model", name=self._model_name())
                if self.mode == "backtest":
                    await self._backtest()
                else:
                    await self._check_contract_spec()
                    outcome = await self._live_loop()
                dispatch.cancel()
                try:
                    await dispatch
                except asyncio.CancelledError:
                    pass
        except Exception as e:
            if self.mode == "live" and not self._stop.is_set() and (
                    isinstance(e, (ConnectionError, OSError, asyncio.TimeoutError))
                    or type(e).__module__.startswith("websockets")):
                self._log(f"Connection error: {e}", "warn")
                return "dropped"
            self._log(f"Engine error: {e}", "error")
            self._emit("stopped", reason=str(e))
        return outcome

    def _model_name(self) -> str:
        return {"3.0": "TimesFM 3.0", "2.5": "TimesFM 2.5"}.get(self.forecaster.kind, "Statistical")

    async def _dispatcher(self):
        try:
            async for raw in self.ws:
                if self._stop.is_set():
                    break
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                rid = data.get("req_id")
                if rid is not None and rid in self._pending:
                    fut = self._pending.pop(rid)
                    if not fut.done():
                        fut.set_result(data)
                else:
                    await self._tick_queue.put(data)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._log(f"WebSocket closed: {e}", "warn")
        else:
            if not self._stop.is_set():
                self._log(f"WebSocket closed by server (code {self.ws.close_code}"
                          f"{', ' + self.ws.close_reason if self.ws.close_reason else ''})", "warn")
        finally:
            err = ConnectionError("WebSocket dispatcher shut down")
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(err)
            self._pending.clear()
            await self._tick_queue.put({"_closed": True})

    async def _send(self, req: dict, timeout: float = 15.0) -> dict:
        self._req_id += 1
        rid = self._req_id
        req["req_id"] = rid
        fut = self.loop.create_future()
        self._pending[rid] = fut
        await self.ws.send(json.dumps(req))
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise asyncio.TimeoutError(f"No response for req_id={rid} after {timeout}s")

    async def _authenticate(self):
        r = await self._send({"balance": 1, "subscribe": 1})
        if "error" in r:
            raise Exception(f"Balance request failed: {r['error'].get('message')}")
        self.balance = float(r["balance"]["balance"])
        self.currency = r["balance"].get("currency", "USD") or "USD"
        self._emit("balance", value=self.balance)
        self._log(f"Connected | Account {self.config.account_id} | "
                  f"Balance {self.balance:.2f} {self.currency}")

    # ── market data ──
    async def _candles(self, symbol: str, gran: int, count: int, end="latest") -> Optional[Candles]:
        if symbol in self._bad_symbols:
            return None
        try:
            r = await self._send({"ticks_history": symbol, "adjust_start_time": 1,
                                  "count": int(min(count, 5000)), "end": end,
                                  "style": "candles", "granularity": gran})
        except Exception as e:
            self._log(f"Candles {symbol}/{gran}s failed: {e}", "warn")
            return None
        if "error" in r:
            self._log(f"Candles {symbol}: {r['error'].get('message')} — skipping symbol", "warn")
            if symbol != self.config.symbol:
                self._bad_symbols.add(symbol)
            return None
        rows = r.get("candles") or []
        return Candles.from_api(rows) if rows else None

    async def _fetch_all(self, m5_count: int, m15_count: int, h1_count: int):
        cfg = self.config
        aux_syms = []
        if cfg.use_s2_macro:
            aux_syms = [SILVER_SYMBOL] + list(DXY_WEIGHTS)
        tasks = [self._candles(cfg.symbol, 300, m5_count)]
        if cfg.use_s4_mtf:
            tasks += [self._candles(cfg.symbol, 900, m15_count),
                      self._candles(cfg.symbol, 3600, h1_count)]
        tasks += [self._candles(s, 300, m5_count) for s in aux_syms]
        res = await asyncio.gather(*tasks)
        m5 = res[0]
        m15 = h1 = None
        i = 1
        if cfg.use_s4_mtf:
            m15, h1 = res[1], res[2]
            i = 3
        aux = {s: c for s, c in zip(aux_syms, res[i:]) if c is not None}
        return m5, m15, h1, aux

    async def _check_contract_spec(self):
        cfg = self.config
        try:
            r = await self._send({"contracts_for": cfg.symbol})
        except Exception as e:
            self._log(f"contracts_for failed ({e}) — continuing with configured duration", "warn")
            return
        if "error" in r:
            self._log(f"contracts_for: {r['error'].get('message')}", "warn")
            return
        avail = r.get("contracts_for", {}).get("available", [])
        want = ("CALL",) if cfg.contract_mode == "RISE_FALL" else ("MULTUP",)
        found = [c for c in avail if c.get("contract_type") in want]
        if not found:
            self._log(f"⚠️  {want[0]} not listed for {cfg.symbol} on this account — trades may be rejected", "warn")
            return
        spans = sorted({f"{c.get('min_contract_duration','?')}–{c.get('max_contract_duration','?')}"
                        for c in found})
        self._log(f"Contract spec {cfg.symbol} {want[0]}: durations {', '.join(spans)}")
        if cfg.contract_mode == "RISE_FALL":
            mins = [self._dur_to_min(c.get("min_contract_duration")) for c in found]
            mins = [m for m in mins if m is not None]
            if mins and cfg.duration_min < min(mins):
                cfg.duration_min = int(min(mins))
                self._log(f"Duration raised to broker minimum: {cfg.duration_min} min", "warn")

    @staticmethod
    def _dur_to_min(s) -> Optional[float]:
        if not s or not isinstance(s, str):
            return None
        unit = s[-1]
        try:
            v = float(s[:-1])
        except ValueError:
            return None
        return {"s": v / 60, "m": v, "h": v * 60, "d": v * 1440, "t": 0}.get(unit)

    # ── day bookkeeping / limits ──
    def _roll_day(self, epoch: float):
        key = datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")
        if key != self.day_key:
            if self.day_key is not None and self.mode == "live":
                self._log(f"New UTC day {key} — daily counters reset")
            self.day_key = key
            self.daily_pnl = 0.0
            self.day_trades = 0
            self.consecutive_losses = 0
            self.recovery_step = 0

    def _limit_hit(self) -> Optional[str]:
        c = self.config
        if self.daily_pnl <= -c.max_daily_loss:
            return f"daily loss limit ({self.daily_pnl:+.2f})"
        if self.daily_pnl >= c.max_daily_profit:
            return f"daily profit target ({self.daily_pnl:+.2f})"
        if self.consecutive_losses >= c.max_consecutive_losses:
            return f"{self.consecutive_losses} consecutive losses"
        if self.day_trades >= c.max_trades_per_day:
            return f"max trades/day ({self.day_trades})"
        return None

    def _stake_for(self, dec: Decision, balance: float) -> float:
        c = self.config
        if c.use_s5_sizing:
            stake = balance * c.risk_pct / 100.0 * dec.size_factor
        else:
            stake = c.base_stake
        if c.use_martingale and self.recovery_step > 0:
            stake *= c.martingale_multiplier ** self.recovery_step
        return round(clamp(stake, c.min_stake, c.max_stake), 2)

    def _record(self, profit: float, balance: float, direction: str, stake: float, when=None):
        self.total_trades += 1
        self.day_trades += 1
        self.daily_pnl += profit
        self.balance = balance
        c = self.config
        if profit > 0:
            self.wins += 1
            self.consecutive_losses = 0
            self.recovery_step = 0
        else:
            self.losses += 1
            self.consecutive_losses += 1
            if c.use_martingale and self.recovery_step < c.max_martingale_steps:
                self.recovery_step += 1
            else:
                self.recovery_step = 0
        wr = self.wins / self.total_trades * 100
        self._emit("result", profit=profit, is_won=profit > 0, balance=balance,
                   daily_pnl=self.daily_pnl, total_trades=self.total_trades,
                   wins=self.wins, losses=self.losses, win_rate=wr,
                   direction=direction, stake=stake, when=when)

    # ── LIVE LOOP ──
    async def _live_loop(self):
        cfg = self.config
        await self.ws.send(json.dumps({"ticks": cfg.symbol, "subscribe": 1}))
        need5 = cfg.ctx_m5 + 10
        need15 = cfg.ctx_m15 + 10
        need1h = cfg.ctx_h1 + 10

        if self._orphan is not None:
            await self._resume_orphan()
            if self._orphan is not None:
                return "dropped"

        # initial analysis (display only — never trade a stale bar)
        self._emit("status", text="ANALYSING", color=WARN)
        await self._analyse(need5, need15, need1h, allow_trade=False)

        last_bar = None
        cooldown_until_bar = self._cooldown_until_bar   # survives reconnects
        quiet_logged = False
        self._emit("status", text="LIVE", color=GREEN)
        while not self._stop.is_set():
            try:
                data = await asyncio.wait_for(self._tick_queue.get(), timeout=30)
            except asyncio.TimeoutError:
                if not quiet_logged:
                    self._log("No ticks for 30s — market may be closed (gold: Sun 23:00 → Fri 21:00 UTC). Waiting…", "warn")
                    self._emit("status", text="MARKET CLOSED?", color=MUTED)
                    quiet_logged = True
                continue
            if data.get("_closed"):
                if self._stop.is_set():
                    break
                self._log("⛔ Feed closed.", "error")
                return "dropped"
            if "balance" in data and isinstance(data["balance"], dict):
                self.balance = float(data["balance"].get("balance", self.balance))
                self._emit("balance", value=self.balance)
                continue
            if "tick" not in data:
                continue
            tick = data["tick"]
            if quiet_logged:
                quiet_logged = False
                self._emit("status", text="LIVE", color=GREEN)
            price = float(tick["quote"])
            epoch = int(tick["epoch"])
            self._emit("tick", price=price)
            self._roll_day(epoch)

            bar = epoch // 300
            if last_bar is None:
                last_bar = bar
                continue
            if bar == last_bar:
                continue
            last_bar = bar

            # ── an M5 bar just closed ──
            reason = self._limit_hit()
            if reason:
                self._log(f"🛑 Stopping for today: {reason}", "warn")
                self._emit("status", text="LIMIT HIT", color=WARN)
                break
            if bar < cooldown_until_bar:
                continue
            await asyncio.sleep(2)            # let the broker finalise the candle
            traded = await self._analyse(need5, need15, need1h, allow_trade=True)
            if traded:
                cooldown_until_bar = (int(utc_now()) // 300) + cfg.cooldown_bars + 1
                self._cooldown_until_bar = cooldown_until_bar

        self._log("🛑 Trading loop ended.")
        self._emit("status", text="STOPPED", color=RED)
        self._emit("stopped", reason="Normal exit")
        return "done"

    async def _analyse(self, n5, n15, n1h, allow_trade: bool) -> bool:
        m5, m15, h1, aux = await self._fetch_all(n5, n15, n1h)
        if m5 is None:
            self._log("No gold candles returned — is the market open?", "warn")
            return False
        now = utc_now()
        m5 = m5.closed_until(now, 300)
        if m15 is not None:
            m15 = m15.closed_until(now, 900)
        if h1 is not None:
            h1 = h1.closed_until(now, 3600)
        t0 = time.monotonic()
        dec: Decision = await self.loop.run_in_executor(
            None, self.strategy.evaluate, now, m5, m15, h1, aux, self.events)
        dt = time.monotonic() - t0
        self._emit("decision", dec=dec, aux=sorted(aux.keys()))
        comp = "  ".join(f"{k}:{v:+.2f}" for k, v in dec.components.items())
        if dec.direction == "FLAT":
            why = "; ".join(dec.reasons[:2]) or "no edge"
            self._log(f"⏭  FLAT  conf {dec.confidence:.1f}  P↑ {dec.prob_up:.2f}  [{comp}]  — {why}  ({dt:.1f}s)", "muted")
            return False
        self._log(f"🎯 {dec.direction}  conf {dec.confidence:.1f}%  P↑ {dec.prob_up:.2f}  "
                  f"regime {dec.regime}  [{comp}]  ({dt:.1f}s)", "signal")
        for r in dec.reasons:
            self._log(f"    · {r}", "muted")
        if not allow_trade:
            self._log("    (initial analysis — waiting for next bar close before trading)", "muted")
            return False
        stake = self._stake_for(dec, self.balance)
        self._emit("signal", direction=dec.direction, confidence=dec.confidence,
                   prob=dec.prob_up, stake=stake, regime=dec.regime)
        self._emit("status", text="PLACING TRADE", color=ACCENT)
        if self.config.contract_mode == "MULTIPLIER":
            res = await self._trade_multiplier(dec, stake)
        else:
            res = await self._trade_rise_fall(dec, stake)
        if res:
            self._report(res, dec.direction, stake)
        elif self._orphan is not None:
            self._orphan.update(direction=dec.direction, stake=stake)
            return True                       # a contract was opened: apply cooldown
        self._emit("status", text="LIVE", color=GREEN)
        return res is not None

    def _report(self, res: dict, direction: str, stake: float):
        self._record(res["profit"], res["balance"], direction, stake)
        lvl = "win" if res["profit"] > 0 else "loss"
        self._log(f"{'✅ WIN' if res['profit'] > 0 else '❌ LOSS'}  {res['profit']:+.2f}  |  "
                  f"balance {res['balance']:.2f}  |  day P&L {self.daily_pnl:+.2f}", lvl)

    async def _resume_orphan(self):
        o, self._orphan = self._orphan, None
        now = time.monotonic()
        self._log(f"🔁 Resuming tracking of contract #{o['cid']}")
        sell_after = max(o["sell_at"] - now, 0.1) if o["sell_at"] else None
        res = await self._wait_result(o["cid"], max(o["end"] - now, 60.0), sell_after=sell_after)
        if self._orphan is not None:          # dropped again — carry the trade details over
            self._orphan.update({k: o[k] for k in ("direction", "stake") if k in o})
        elif res and "direction" in o:
            self._report(res, o["direction"], o["stake"])

    async def _buy(self, proposal: dict, stake: float) -> Optional[Tuple[str, float]]:
        for attempt in range(self.config.max_retries):
            try:
                pr = await self._send(dict(proposal))
                if "error" in pr:
                    self._log(f"Proposal error: {pr['error'].get('message')}", "warn")
                    await asyncio.sleep(0.3)
                    continue
                p = pr["proposal"]
                payout = float(p.get("payout", 0) or 0)
                if self.config.contract_mode == "RISE_FALL" and payout:
                    ratio = (payout - stake) / stake
                    if ratio < self.config.min_payout_ratio:
                        self._log(f"Payout ratio {ratio:.0%} below minimum "
                                  f"{self.config.min_payout_ratio:.0%} — skipping", "warn")
                        return None
                br = await self._send({"buy": p["id"], "price": stake})
                if "error" in br:
                    self._log(f"Buy error: {br['error'].get('message')}", "warn")
                    await asyncio.sleep(0.3)
                    continue
                cid = br["buy"]["contract_id"]
                self._log(f"📋 Contract #{cid} opened | stake {stake:.2f} | payout {payout:.2f}")
                return cid, payout
            except Exception as e:
                self._log(f"Trade attempt {attempt + 1} failed: {e}", "warn")
                await asyncio.sleep(0.3)
        return None

    async def _trade_rise_fall(self, dec: Decision, stake: float) -> Optional[dict]:
        c = self.config
        prop = {"proposal": 1, "amount": stake, "basis": "stake",
                "contract_type": "CALL" if dec.direction == "RISE" else "PUT",
                "currency": self.currency, "duration": c.duration_min,
                "duration_unit": "m", "underlying_symbol": c.symbol}
        got = await self._buy(prop, stake)
        if not got:
            return None
        return await self._wait_result(got[0], c.duration_min * 60 + 120, sell_after=None)

    async def _trade_multiplier(self, dec: Decision, stake: float) -> Optional[dict]:
        c = self.config
        px = max(dec.price, 1e-9)
        sl = round(min(stake * 0.95, stake * c.multiplier * dec.sl_dist / px), 2)
        tp = round(stake * c.multiplier * dec.tp_dist / px, 2)
        prop = {"proposal": 1, "amount": stake, "basis": "stake",
                "contract_type": "MULTUP" if dec.direction == "RISE" else "MULTDOWN",
                "currency": self.currency, "multiplier": c.multiplier,
                "underlying_symbol": c.symbol}
        if sl >= 0.1 and tp >= 0.1:
            prop["limit_order"] = {"stop_loss": sl, "take_profit": tp}
            self._log(f"S5 quantile exits: SL ${sl:.2f} (p{'10' if dec.direction=='RISE' else '90'}) | TP ${tp:.2f}")
        got = await self._buy(prop, stake)
        if not got:
            return None
        return await self._wait_result(got[0], c.max_hold_min * 60 + 300, sell_after=c.max_hold_min * 60)

    async def _wait_result(self, cid, timeout: float, sell_after: Optional[float]) -> Optional[dict]:
        await self.ws.send(json.dumps({"proposal_open_contract": 1, "contract_id": cid, "subscribe": 1}))
        start = time.monotonic()
        sold_requested = False
        while time.monotonic() - start < timeout and not self._stop.is_set():
            if sell_after and not sold_requested and time.monotonic() - start > sell_after:
                self._log(f"⏱  Max hold reached — closing #{cid}")
                asyncio.ensure_future(self.ws.send(json.dumps({"sell": cid, "price": 0})))
                sold_requested = True
            try:
                data = await asyncio.wait_for(self._tick_queue.get(), timeout=5)
            except asyncio.TimeoutError:
                continue
            if data.get("_closed"):
                # remember the open contract so the next connection can resume tracking it
                self._orphan = {"cid": cid, "end": start + timeout,
                                "sell_at": None if (sold_requested or not sell_after) else start + sell_after}
                self._log(f"Feed lost while #{cid} is open — will resume after reconnect", "warn")
                await self._tick_queue.put(data)     # let the live loop see the close too
                return None
            if "tick" in data:
                self._emit("tick", price=float(data["tick"]["quote"]))
                continue
            poc = data.get("proposal_open_contract")
            if poc and str(poc.get("contract_id")) == str(cid):
                if poc.get("is_sold"):
                    profit = float(poc.get("profit", 0))
                    bal = float(poc.get("balance_after") or (self.balance + profit))
                    asyncio.ensure_future(self.ws.send(json.dumps({"forget_all": "proposal_open_contract"})))
                    return {"profit": profit, "balance": bal}
                elif poc.get("profit") is not None:
                    self._emit("open_pnl", value=float(poc["profit"]))
        self._log(f"Contract #{cid} result timeout — check Deriv statement", "warn")
        return None

    # ── BACKTEST ──
    async def _backtest(self):
        c = self.config
        dur = self.strategy.dur_bars
        n5 = min(5000, c.bt_bars + c.ctx_m5 + dur + 5)
        n15 = min(5000, n5 // 3 + c.ctx_m15 + 10)
        n1h = min(5000, n5 // 12 + c.ctx_h1 + 10)
        self._emit("status", text="DOWNLOADING", color=WARN)
        self._log(f"Backtest: downloading {n5} M5, {n15} M15, {n1h} H1 candles + macro series…")
        m5, m15, h1, aux = await self._fetch_all(n5, n15, n1h)
        if m5 is None or len(m5) < c.ctx_m5 // 2 + dur + 50:
            self._log("Not enough history for backtest.", "error")
            self._emit("stopped", reason="No data")
            return
        self._log(f"Got {len(m5)} M5 bars ({datetime.utcfromtimestamp(m5.epoch[0]):%Y-%m-%d} → "
                  f"{datetime.utcfromtimestamp(m5.epoch[-1]):%Y-%m-%d}); macro: {', '.join(sorted(aux)) or 'none'}")
        self._emit("status", text="BACKTESTING", color=ACCENT)

        start = min(c.ctx_m5, len(m5) // 3)
        end = len(m5) - dur
        bal = c.bt_start_balance
        peak, max_dd = bal, 0.0
        self.balance = bal
        equity = []
        next_ok = start
        t_idx = list(range(start, end, max(1, c.bt_step)))
        stopped_for_day = None
        flat_reasons: Dict[str, int] = {}
        t0 = time.monotonic()

        for n, i in enumerate(t_idx):
            if self._stop.is_set():
                break
            if i < next_ok:
                continue
            now = float(m5.epoch[i] + 300)
            self._roll_day(now)
            if stopped_for_day == self.day_key:
                continue
            if self._limit_hit():
                stopped_for_day = self.day_key
                continue
            m5s = m5.slice(max(0, i + 1 - c.ctx_m5 - 2), i + 1)
            m15s = m15.closed_until(now, 900) if m15 is not None else None
            h1s = h1.closed_until(now, 3600) if h1 is not None else None
            auxs = {s: cc.closed_until(now, 300) for s, cc in aux.items()}
            dec = await self.loop.run_in_executor(
                None, self.strategy.evaluate, now, m5s, m15s, h1s, auxs, self.events)
            if n % 10 == 0:
                self._emit("progress", value=n + 1, total=len(t_idx), eta=(time.monotonic() - t0) / (n + 1) * (len(t_idx) - n))
                self._emit("decision", dec=dec, aux=sorted(aux))
            if dec.direction == "FLAT":
                k = re.split(r"\s[\d<(]|:", dec.reasons[0])[0].strip() if dec.reasons else "none"
                flat_reasons[k] = flat_reasons.get(k, 0) + 1
                continue
            entry = float(m5.close[i])
            exit_ = float(m5.close[i + dur])
            won = exit_ > entry if dec.direction == "RISE" else exit_ < entry
            stake = self._stake_for(dec, bal)
            profit = round(stake * c.bt_payout, 2) if won else -stake
            bal += profit
            peak = max(peak, bal)
            max_dd = max(max_dd, peak - bal)
            equity.append(bal)
            self._record(profit, bal, dec.direction, stake,
                         when=datetime.utcfromtimestamp(now).strftime("%m-%d %H:%M"))
            next_ok = i + dur + c.cooldown_bars

        # ── summary ──
        n_tr = self.total_trades
        days = max(1.0, (m5.epoch[min(end, len(m5) - 1)] - m5.epoch[start]) / 86400 * 5 / 7)
        wr = self.wins / n_tr if n_tr else 0.0
        be = 1 / (1 + c.bt_payout)
        pnl = bal - c.bt_start_balance
        exp_per_unit = wr * c.bt_payout - (1 - wr)
        tpd = n_tr / days
        lines = [
            "════════ BACKTEST SUMMARY ════════",
            f"Model: {self._model_name()} | bars tested: {len(t_idx)} (step {c.bt_step}) | "
            f"{days:.1f} trading days | {time.monotonic() - t0:.0f}s",
            f"Trades {n_tr}  ({tpd:.1f}/day) | Wins {self.wins} | Losses {self.losses} | "
            f"Win rate {wr:.1%}  (break-even {be:.1%} at {c.bt_payout:.0%} payout)",
            f"Net P&L {pnl:+.2f} on {c.bt_start_balance:.0f} start | Max drawdown {max_dd:.2f}",
            f"Expectancy per $1 staked: {exp_per_unit:+.3f}",
        ]
        if n_tr and exp_per_unit > 0 and tpd > 0:
            need = 500 / (exp_per_unit * tpd)
            lines.append(f"To average $500/day at this edge you'd need ≈ ${need:,.0f} per trade "
                         f"(≈ ${need * 100 / max(c.risk_pct, 0.1):,.0f} account at {c.risk_pct}% risk).")
        elif n_tr:
            lines.append("No positive edge on this sample — do NOT trade this configuration live.")
        if flat_reasons:
            top = sorted(flat_reasons.items(), key=lambda x: -x[1])[:5]
            lines.append("Top skip reasons: " + ", ".join(f"{k} ×{v}" for k, v in top))
        for ln in lines:
            self._log(ln, "win" if exp_per_unit > 0 and n_tr else "warn")
        self._emit("status", text="BACKTEST DONE", color=GREEN)
        self._emit("stopped", reason="Normal exit")


# ─────────────────────────────────────────────
#  CANVAS WIDGETS
# ─────────────────────────────────────────────
class Sparkline(tk.Canvas):
    def __init__(self, parent, width=400, height=70, color=ACCENT, **kw):
        super().__init__(parent, width=width, height=height, bg=CARD, highlightthickness=0, **kw)
        self.color = color
        self.data: deque = deque(maxlen=300)
        self.bind("<Configure>", lambda e: self._redraw())

    def add(self, v: float):
        self.data.append(v)
        self._redraw()

    def clear(self):
        self.data.clear()
        self._redraw()

    def _redraw(self):
        self.delete("all")
        pts = list(self.data)
        w, h = self.winfo_width() or 400, self.winfo_height() or 70
        if len(pts) < 2:
            return
        lo, hi = min(pts + [0]), max(pts + [0])
        rng = hi - lo or 1
        pad = 4
        y = lambda v: h - pad - (v - lo) / rng * (h - 2 * pad)
        x = lambda i: pad + i / (len(pts) - 1) * (w - 2 * pad)
        self.create_line(pad, y(0), w - pad, y(0), fill=BORD, dash=(2, 3))
        coords = []
        for i, v in enumerate(pts):
            coords += [x(i), y(v)]
        col = GREEN if pts[-1] >= 0 else RED
        self.create_line(*coords, fill=col, width=1.6)


class ForecastChart(tk.Canvas):
    """Last 60 M5 closes + TimesFM p10–p90 fan + p50 path; marks the expiry bar."""

    def __init__(self, parent, width=400, height=170, **kw):
        super().__init__(parent, width=width, height=height, bg=CARD, highlightthickness=0, **kw)
        self.chart = None
        self.bind("<Configure>", lambda e: self._redraw())

    def set(self, chart: Optional[dict]):
        self.chart = chart
        self._redraw()

    def _redraw(self):
        self.delete("all")
        w, h = self.winfo_width() or 400, self.winfo_height() or 170
        c = self.chart
        if not c:
            self.create_text(w / 2, h / 2, text="Forecast appears after first analysis",
                             fill=MUTED, font=FONT_SM)
            return
        hist, p10, p50, p90 = c["hist"], c["p10"], c["p50"], c["p90"]
        n_h, n_f = len(hist), len(p50)
        allv = hist + p10 + p90
        lo, hi = min(allv), max(allv)
        rng = hi - lo or 1
        pl, pr, pt, pb = 8, 62, 10, 14
        total = n_h + n_f
        X = lambda i: pl + i / (total - 1) * (w - pl - pr)
        Y = lambda v: h - pb - (v - lo) / rng * (h - pt - pb)
        # fan
        poly = [X(n_h - 1), Y(hist[-1])]
        for j, v in enumerate(p90):
            poly += [X(n_h + j), Y(v)]
        for j in range(n_f - 1, -1, -1):
            poly += [X(n_h + j), Y(p10[j])]
        poly += [X(n_h - 1), Y(hist[-1])]
        self.create_polygon(poly, fill=GOLD, outline="", stipple="gray25")
        # history
        coords = []
        for i, v in enumerate(hist):
            coords += [X(i), Y(v)]
        self.create_line(*coords, fill=TEXT, width=1.4)
        # median
        mc = [X(n_h - 1), Y(hist[-1])]
        for j, v in enumerate(p50):
            mc += [X(n_h + j), Y(v)]
        self.create_line(*mc, fill=GOLD, width=2)
        # last price line + expiry marker
        self.create_line(pl, Y(hist[-1]), w - pr, Y(hist[-1]), fill=BORD, dash=(2, 3))
        kx = X(n_h + c.get("k", 0))
        self.create_line(kx, pt, kx, h - pb, fill=ACCENT, dash=(3, 3))
        self.create_text(kx, h - 2, text="expiry", fill=ACCENT, font=("Segoe UI", 7), anchor="s")
        for v, col in ((hi, MUTED), (lo, MUTED), (hist[-1], TEXT)):
            self.create_text(w - pr + 4, Y(v), text=f"{v:.2f}", fill=col,
                             font=("Consolas", 8), anchor="w")


class SetupBars(tk.Canvas):
    """Diverging bars for each setup score in [-1, 1] + the ensemble score."""
    ORDER = ["Prob", "S1 Band", "S2 Macro", "S4 MTF"]

    def __init__(self, parent, width=400, height=130, **kw):
        super().__init__(parent, width=width, height=height, bg=CARD, highlightthickness=0, **kw)
        self.comps: Dict[str, float] = {}
        self.score = 0.0
        self.blocked = ""
        self.bind("<Configure>", lambda e: self._redraw())

    def set(self, comps, score, blocked=""):
        self.comps, self.score, self.blocked = comps, score, blocked
        self._redraw()

    def _redraw(self):
        self.delete("all")
        w = self.winfo_width() or 400
        rows = [(n, self.comps.get(n)) for n in self.ORDER] + [("ENSEMBLE", self.score if self.comps else None)]
        lw, cx_pad = 78, 50
        mid = lw + (w - lw - cx_pad) / 2
        half = (w - lw - cx_pad) / 2 - 4
        for r, (name, v) in enumerate(rows):
            y = 10 + r * 20
            bold = name == "ENSEMBLE"
            self.create_text(4, y + 6, text=name, anchor="w", fill=TEXT if bold else MUTED,
                             font=("Segoe UI", 8, "bold" if bold else "normal"))
            self.create_rectangle(lw, y, w - cx_pad, y + 12, fill=PANEL, outline="")
            self.create_line(mid, y - 1, mid, y + 13, fill=BORD)
            if v is None:
                self.create_text(mid, y + 6, text="off", fill=MUTED, font=("Segoe UI", 7))
                continue
            col = GREEN if v >= 0 else RED
            self.create_rectangle(mid, y + 2, mid + v * half, y + 10, fill=col, outline="")
            self.create_text(w - cx_pad + 6, y + 6, text=f"{v:+.2f}", anchor="w", fill=col,
                             font=("Consolas", 8))
        if self.blocked:
            self.create_text(4, 10 + len(rows) * 20 + 2, text=f"S3 ⛔ {self.blocked}", anchor="w",
                             fill=WARN, font=("Segoe UI", 8))


# ─────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────
class GoldBotApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("TimesFM Gold Bot — XAUUSD Rise/Fall")
        self.geometry("1320x860")
        self.minsize(1100, 760)
        self.configure(bg=DARK)

        self._demo_config = GoldBotConfig.load("gold_config_demo.json")
        self._real_config = GoldBotConfig.load("gold_config_real.json")
        self._real_config.use_real_account = True
        self.config_obj = self._demo_config
        self.msg_queue: queue.Queue = queue.Queue()
        self._engine: Optional[TradingEngine] = None
        self._mode = "live"

        self._build_ui()
        self._populate_fields()
        self._poll_queue()

    # ─── layout helpers ───
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TNotebook", background=DARK, borderwidth=0)
        s.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=[8, 5], font=FONT_SM)
        s.map("TNotebook.Tab", background=[("selected", CARD)], foreground=[("selected", TEXT)])
        s.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=TEXT, arrowcolor=GOLD)
        s.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", TEXT)])
        s.configure("Accent.TButton", background=GOLD, foreground=DARK,
                    font=("Segoe UI", 10, "bold"), padding=[12, 6], relief="flat")
        s.configure("Ghost.TButton", background=CARD, foreground=ACCENT,
                    font=("Segoe UI", 10, "bold"), padding=[12, 6], relief="flat")
        s.configure("Stop.TButton", background=RED, foreground="white",
                    font=("Segoe UI", 10, "bold"), padding=[12, 6], relief="flat")
        s.configure("TProgressbar", troughcolor=BORD, background=GOLD, borderwidth=0, thickness=4)
        s.configure("Treeview", background=PANEL, foreground=TEXT, fieldbackground=PANEL,
                    rowheight=22, font=FONT_MONO, borderwidth=0)
        s.configure("Treeview.Heading", background=CARD, foreground=MUTED,
                    font=("Segoe UI", 8, "bold"), relief="flat")

    def _card(self, parent, title="") -> tk.Frame:
        outer = tk.Frame(parent, bg=CARD)
        outer.pack(fill="x", pady=(0, 6))
        inner = tk.Frame(outer, bg=CARD)
        inner.pack(fill="both", padx=10, pady=8)
        if title:
            tk.Label(inner, text=title.upper(), bg=CARD, fg=MUTED,
                     font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 4))
        return inner

    def _row(self, parent, label, factory, **kw):
        f = tk.Frame(parent, bg=CARD)
        f.pack(fill="x", pady=1)
        tk.Label(f, text=label, bg=CARD, fg=MUTED, font=FONT_SM, width=20, anchor="w").pack(side="left")
        w = factory(f, **kw)
        w.pack(side="right", fill="x", expand=True)
        return w

    def _entry(self, parent, textvariable, show=""):
        return tk.Entry(parent, textvariable=textvariable, width=16, bg=PANEL, fg=TEXT, insertbackground=GOLD,
                        relief="flat", font=FONT_MONO, bd=0, highlightthickness=1,
                        highlightcolor=GOLD, highlightbackground=BORD, show=show)

    def _spin(self, parent, textvariable, from_, to, increment=1):
        return tk.Spinbox(parent, textvariable=textvariable, from_=from_, to=to, increment=increment, width=10,
                          bg=PANEL, fg=TEXT, buttonbackground=BORD, insertbackground=GOLD,
                          relief="flat", font=FONT_MONO, bd=0, highlightthickness=1,
                          highlightcolor=GOLD, highlightbackground=BORD)

    def _combo(self, parent, textvariable, values):
        return ttk.Combobox(parent, textvariable=textvariable, values=values, width=13,
                            state="readonly", font=FONT_MONO)

    def _check(self, parent, text, var, fg=TEXT):
        cb = tk.Checkbutton(parent, text=text, variable=var, bg=CARD, fg=fg, activebackground=CARD,
                            activeforeground=fg, selectcolor=DARK, font=FONT_SM, anchor="w")
        cb.pack(fill="x", anchor="w")
        return cb

    # ─── build ───
    def _build_ui(self):
        self._style()
        self._build_topbar()
        body = tk.Frame(self, bg=DARK)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        body.columnconfigure(0, weight=0, minsize=350)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        left = tk.Frame(body, bg=DARK, width=350)
        right = tk.Frame(body, bg=DARK)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right.grid(row=0, column=1, sticky="nsew")
        self._build_config_panel(left)
        self._build_dashboard(right)

    def _build_topbar(self):
        bar = tk.Frame(self, bg=PANEL, height=52)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Label(bar, text="◆  TimesFM Gold Bot", bg=PANEL, fg=GOLD,
                 font=("Segoe UI", 14, "bold")).pack(side="left", padx=16)
        tk.Label(bar, text="XAUUSD · Rise/Fall", bg=PANEL, fg=MUTED, font=FONT_SM).pack(side="left")

        self._status_var = tk.StringVar(value="  OFFLINE  ")
        self._status_lbl = tk.Label(bar, textvariable=self._status_var, bg=MUTED, fg="white",
                                    font=("Segoe UI", 9, "bold"), padx=10, pady=3)
        self._status_lbl.pack(side="right", padx=16)
        self._balance_var = tk.StringVar(value="—")
        tk.Label(bar, textvariable=self._balance_var, bg=PANEL, fg=GREEN,
                 font=("Segoe UI", 12, "bold")).pack(side="right", padx=(4, 18))
        tk.Label(bar, text="Balance", bg=PANEL, fg=MUTED, font=FONT_SM).pack(side="right")
        self._price_var = tk.StringVar(value="—")
        tk.Label(bar, textvariable=self._price_var, bg=PANEL, fg=GOLD,
                 font=("Consolas", 12, "bold")).pack(side="right", padx=(4, 18))
        tk.Label(bar, text="XAUUSD", bg=PANEL, fg=MUTED, font=FONT_SM).pack(side="right")
        self._model_var = tk.StringVar(value="model: —")
        tk.Label(bar, textvariable=self._model_var, bg=PANEL, fg=ACCENT, font=FONT_SM).pack(side="right", padx=16)

    def _build_config_panel(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)
        tabs = {}
        for name in ("Connect", "Setups", "Model", "Risk", "Events"):
            fr = tk.Frame(nb, bg=DARK)
            nb.add(fr, text=f" {name} ")
            tabs[name] = fr
        self._build_conn_tab(tabs["Connect"])
        self._build_setups_tab(tabs["Setups"])
        self._build_model_tab(tabs["Model"])
        self._build_risk_tab(tabs["Risk"])
        self._build_events_tab(tabs["Events"])

        btns = tk.Frame(parent, bg=DARK)
        btns.pack(fill="x", pady=(8, 0))
        self._start_btn = ttk.Button(btns, text="▶ Start", style="Accent.TButton", command=self._start_bot)
        self._start_btn.pack(side="left", padx=(0, 4))
        self._bt_btn = ttk.Button(btns, text="⟲ Backtest", style="Ghost.TButton", command=self._start_backtest)
        self._bt_btn.pack(side="left", padx=4)
        self._stop_btn = ttk.Button(btns, text="■ Stop", style="Stop.TButton",
                                    command=self._stop_bot, state="disabled")
        self._stop_btn.pack(side="left", padx=4)
        ttk.Button(btns, text="Save", command=self._save_config).pack(side="right")

    def _build_conn_tab(self, p):
        f = self._card(p, "API credentials")
        self._v_token, self._v_app_id, self._v_account_id = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self._v_account_type = tk.StringVar(value="demo")
        self._row(f, "Access Token", self._entry, textvariable=self._v_token, show="•")
        self._row(f, "App ID", self._entry, textvariable=self._v_app_id)
        self._row(f, "Account ID", self._entry, textvariable=self._v_account_id)
        rf = tk.Frame(f, bg=CARD)
        rf.pack(anchor="w", pady=(8, 0))
        tk.Label(rf, text="Account:", bg=CARD, fg=MUTED, font=FONT_SM).pack(side="left")
        for txt, val, col in (("Demo", "demo", TEXT), ("Real (⚠ real money!)", "real", WARN)):
            tk.Radiobutton(rf, text=txt, variable=self._v_account_type, value=val,
                           command=self._on_account_switch, bg=CARD, fg=col, selectcolor=DARK,
                           activebackground=CARD, font=FONT_SM).pack(side="left", padx=4)

        f2 = self._card(p, "Contract")
        self._v_symbol, self._v_mode = tk.StringVar(), tk.StringVar()
        self._v_duration, self._v_mult, self._v_hold = tk.IntVar(), tk.IntVar(), tk.IntVar()
        self._v_min_payout = tk.DoubleVar()
        self._row(f2, "Symbol", self._combo, textvariable=self._v_symbol, values=[GOLD_SYMBOL])
        self._row(f2, "Contract", self._combo, textvariable=self._v_mode, values=["RISE_FALL", "MULTIPLIER"])
        self._row(f2, "Rise/Fall expiry (min)", self._spin, textvariable=self._v_duration, from_=1, to=1440)
        self._row(f2, "Min payout ratio", self._spin, textvariable=self._v_min_payout, from_=0.3, to=1.0, increment=0.05)
        self._row(f2, "Multiplier (x)", self._spin, textvariable=self._v_mult, from_=1, to=1000)
        self._row(f2, "Mult. max hold (min)", self._spin, textvariable=self._v_hold, from_=5, to=1440, increment=5)
        tk.Label(f2, text="The bot reads the broker's allowed durations on start\n"
                          "and raises expiry to the minimum if needed.",
                 bg=CARD, fg=MUTED, font=("Segoe UI", 8), justify="left").pack(anchor="w", pady=(6, 0))

    def _build_setups_tab(self, p):
        f = self._card(p, "The five setups")
        self._v_s1, self._v_s2, self._v_s3 = tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
        self._v_s4, self._v_s5, self._v_req_mtf = tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
        self._check(f, "S1  Quantile-band breakout", self._v_s1, GOLD)
        self._check(f, "S2  Multivariate macro (DXY-proxy + Silver)", self._v_s2, GOLD)
        self._check(f, "S3  Event covariates + blackout", self._v_s3, GOLD)
        self._check(f, "S4  Multi-timeframe confluence (H1/M15/M5)", self._v_s4, GOLD)
        self._check(f, "S5  Quantile-band risk sizing", self._v_s5, GOLD)
        self._check(f, "Require all timeframes to agree", self._v_req_mtf)

        f2 = self._card(p, "Ensemble weights & thresholds")
        self._v_w_prob, self._v_w_band, self._v_w_macro, self._v_w_mtf = (tk.DoubleVar() for _ in range(4))
        self._v_conf, self._v_min_prob = tk.DoubleVar(), tk.DoubleVar()
        self._v_sess_a, self._v_sess_b, self._v_cool = tk.IntVar(), tk.IntVar(), tk.IntVar()
        self._row(f2, "Weight: quantile prob", self._spin, textvariable=self._v_w_prob, from_=0, to=3, increment=0.1)
        self._row(f2, "Weight: S1 band", self._spin, textvariable=self._v_w_band, from_=0, to=3, increment=0.1)
        self._row(f2, "Weight: S2 macro", self._spin, textvariable=self._v_w_macro, from_=0, to=3, increment=0.1)
        self._row(f2, "Weight: S4 MTF", self._spin, textvariable=self._v_w_mtf, from_=0, to=3, increment=0.1)
        self._row(f2, "Min confidence %", self._spin, textvariable=self._v_conf, from_=50, to=99, increment=0.5)
        self._row(f2, "Min directional prob", self._spin, textvariable=self._v_min_prob, from_=0.5, to=0.95, increment=0.01)
        self._row(f2, "Session start (UTC h)", self._spin, textvariable=self._v_sess_a, from_=0, to=23)
        self._row(f2, "Session end (UTC h)", self._spin, textvariable=self._v_sess_b, from_=1, to=24)
        self._row(f2, "Cooldown (M5 bars)", self._spin, textvariable=self._v_cool, from_=0, to=48)

    def _build_model_tab(self, p):
        f = self._card(p, "Forecaster")
        self._v_use_tfm = tk.BooleanVar()
        self._v_model, self._v_device = tk.StringVar(), tk.StringVar()
        self._v_ctx5, self._v_ctx15, self._v_ctx1h, self._v_hzn = (tk.IntVar() for _ in range(4))
        self._check(f, "Use TimesFM neural forecasting", self._v_use_tfm, ACCENT)
        self._row(f, "Model", self._combo, textvariable=self._v_model, values=["auto", "3.0", "2.5", "stats"])
        self._row(f, "Device", self._combo, textvariable=self._v_device, values=["auto", "cpu", "cuda", "mps"])
        self._row(f, "M5 context bars", self._spin, textvariable=self._v_ctx5, from_=128, to=4096, increment=64)
        self._row(f, "M15 context bars", self._spin, textvariable=self._v_ctx15, from_=64, to=2048, increment=64)
        self._row(f, "H1 context bars", self._spin, textvariable=self._v_ctx1h, from_=64, to=2048, increment=64)
        self._row(f, "Horizon (M5 bars)", self._spin, textvariable=self._v_hzn, from_=3, to=96)
        tk.Label(f, text="TimesFM 3.0 weights are non-commercial licence.\n"
                         "Pick 2.5 (Apache-2.0) if that matters to you.",
                 bg=CARD, fg=MUTED, font=("Segoe UI", 8), justify="left").pack(anchor="w", pady=(6, 0))

        f2 = self._card(p, "Backtest")
        self._v_bt_bars, self._v_bt_step = tk.IntVar(), tk.IntVar()
        self._v_bt_payout, self._v_bt_bal = tk.DoubleVar(), tk.DoubleVar()
        self._row(f2, "M5 bars to test", self._spin, textvariable=self._v_bt_bars, from_=200, to=4500, increment=100)
        self._row(f2, "Evaluate every N bars", self._spin, textvariable=self._v_bt_step, from_=1, to=12)
        self._row(f2, "Assumed payout ratio", self._spin, textvariable=self._v_bt_payout, from_=0.5, to=1.0, increment=0.01)
        self._row(f2, "Start balance", self._spin, textvariable=self._v_bt_bal, from_=10, to=1_000_000, increment=100)

    def _build_risk_tab(self, p):
        f = self._card(p, "Stake sizing")
        self._v_risk, self._v_stake, self._v_min_stake, self._v_max_stake = (tk.DoubleVar() for _ in range(4))
        self._v_use_mart = tk.BooleanVar()
        self._v_mart_mul, self._v_mart_steps = tk.DoubleVar(), tk.IntVar()
        self._row(f, "Risk % per trade (S5)", self._spin, textvariable=self._v_risk, from_=0.1, to=10, increment=0.1)
        self._row(f, "Fixed stake (S5 off)", self._spin, textvariable=self._v_stake, from_=0.35, to=10000, increment=0.5)
        self._row(f, "Min stake", self._spin, textvariable=self._v_min_stake, from_=0.35, to=10000, increment=0.5)
        self._row(f, "Max stake", self._spin, textvariable=self._v_max_stake, from_=1, to=100000, increment=5)
        self._check(f, "Martingale recovery (⚠ raises ruin risk)", self._v_use_mart, WARN)
        self._row(f, "Martingale multiplier", self._spin, textvariable=self._v_mart_mul, from_=1.1, to=3, increment=0.1)
        self._row(f, "Max martingale steps", self._spin, textvariable=self._v_mart_steps, from_=1, to=4)

        f2 = self._card(p, "Daily guard-rails")
        self._v_max_loss, self._v_max_profit = tk.DoubleVar(), tk.DoubleVar()
        self._v_max_consec, self._v_max_tpd, self._v_retries = tk.IntVar(), tk.IntVar(), tk.IntVar()
        self._row(f2, "Max daily loss", self._spin, textvariable=self._v_max_loss, from_=1, to=100000, increment=10)
        self._row(f2, "Daily profit target", self._spin, textvariable=self._v_max_profit, from_=1, to=100000, increment=10)
        self._row(f2, "Max consecutive losses", self._spin, textvariable=self._v_max_consec, from_=1, to=20)
        self._row(f2, "Max trades per day", self._spin, textvariable=self._v_max_tpd, from_=1, to=500)
        self._row(f2, "Max order retries", self._spin, textvariable=self._v_retries, from_=1, to=5)

    def _build_events_tab(self, p):
        f = self._card(p, "S3 economic calendar (UTC)")
        self._v_pre, self._v_post = tk.IntVar(), tk.IntVar()
        self._row(f, "Blackout before (min)", self._spin, textvariable=self._v_pre, from_=0, to=240, increment=5)
        self._row(f, "Blackout after (min)", self._spin, textvariable=self._v_post, from_=0, to=240, increment=5)
        tk.Label(f, text="NFP · CPI · FOMC · PCE · GDP · Powell speeches.\n"
                         "Fed to TimesFM 3.0 as known-future covariates and\n"
                         "used as a hard no-trade window.",
                 bg=CARD, fg=MUTED, font=("Segoe UI", 8), justify="left").pack(anchor="w", pady=(4, 4))
        self._events_box = scrolledtext.ScrolledText(f, bg=PANEL, fg=TEXT, font=FONT_MONO, height=16, width=34,
                                                     relief="flat", insertbackground=GOLD, wrap="none")
        self._events_box.pack(fill="both", expand=True)

    def _build_dashboard(self, parent):
        parent.columnconfigure(0, weight=3)
        parent.columnconfigure(1, weight=2)
        parent.rowconfigure(1, weight=1)

        stats = tk.Frame(parent, bg=DARK)
        stats.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self._stat_vars, self._stat_lbls = {}, {}
        defs = [("daily_pnl", "Day P&L", "0.00", GREEN), ("win_rate", "Win rate", "0.0%", ACCENT),
                ("total_trades", "Trades", "0", TEXT), ("wins", "Wins", "0", GREEN),
                ("losses", "Losses", "0", RED), ("open_pnl", "Open P&L", "—", GOLD)]
        for i, (k, label, d, col) in enumerate(defs):
            stats.columnconfigure(i, weight=1)
            c = tk.Frame(stats, bg=CARD)
            c.grid(row=0, column=i, padx=3, sticky="ew", ipady=6)
            tk.Label(c, text=label, bg=CARD, fg=MUTED, font=("Segoe UI", 8)).pack()
            v = tk.StringVar(value=d)
            lb = tk.Label(c, textvariable=v, bg=CARD, fg=col, font=("Segoe UI", 14, "bold"))
            lb.pack()
            self._stat_vars[k], self._stat_lbls[k] = v, lb

        left = tk.Frame(parent, bg=DARK)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        right = tk.Frame(parent, bg=DARK)
        right.grid(row=1, column=1, sticky="nsew", padx=(4, 0))

        # signal card
        sig = tk.Frame(left, bg=CARD)
        sig.pack(fill="x", pady=(0, 6))
        tk.Label(sig, text="LAST DECISION", bg=CARD, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=10, pady=(6, 0))
        row = tk.Frame(sig, bg=CARD)
        row.pack(fill="x", padx=10, pady=(0, 6))
        self._sig_vars = {}
        for name, col in (("Direction", GOLD), ("Confidence", TEXT), ("P(up)", ACCENT),
                          ("Regime", WARN), ("Stake", GREEN)):
            fr = tk.Frame(row, bg=CARD)
            fr.pack(side="left", expand=True, fill="x")
            tk.Label(fr, text=name, bg=CARD, fg=MUTED, font=("Segoe UI", 7)).pack()
            v = tk.StringVar(value="—")
            lb = tk.Label(fr, textvariable=v, bg=CARD, fg=col, font=("Segoe UI", 12, "bold"))
            lb.pack()
            self._sig_vars[name] = (v, lb)

        fc = tk.Frame(left, bg=CARD)
        fc.pack(fill="x", pady=(0, 6))
        tk.Label(fc, text="TIMESFM FORECAST FAN  (p10–p90, p50)", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=10, pady=(6, 0))
        self._fchart = ForecastChart(fc, height=170)
        self._fchart.pack(fill="x", padx=10, pady=(4, 8))

        sb = tk.Frame(left, bg=CARD)
        sb.pack(fill="x", pady=(0, 6))
        tk.Label(sb, text="SETUP SCORES", bg=CARD, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=10, pady=(6, 0))
        self._bars = SetupBars(sb, height=135)
        self._bars.pack(fill="x", padx=10, pady=(4, 6))

        sp = tk.Frame(left, bg=CARD)
        sp.pack(fill="x", pady=(0, 6))
        tk.Label(sp, text="P&L (CUMULATIVE)", bg=CARD, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=10, pady=(6, 0))
        self._spark = Sparkline(sp, height=60)
        self._spark.pack(fill="x", padx=10, pady=(4, 8))

        pg = tk.Frame(left, bg=CARD)
        pg.pack(fill="x")
        self._prog_var = tk.IntVar()
        self._prog_lbl = tk.Label(pg, text="", bg=CARD, fg=MUTED, font=FONT_SM)
        self._prog_lbl.pack(anchor="w", padx=10)
        ttk.Progressbar(pg, variable=self._prog_var, style="TProgressbar").pack(fill="x", padx=10, pady=(2, 8))

        # trade history + log
        tk.Label(right, text="TRADE HISTORY", bg=DARK, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w")
        tf = tk.Frame(right, bg=DARK)
        tf.pack(fill="both", expand=False)
        cols = ("Time", "Dir", "Stake", "Profit", "Balance")
        self._tree = ttk.Treeview(tf, columns=cols, show="headings", height=9, selectmode="none")
        for ccol, wdt in zip(cols, (78, 48, 58, 66, 80)):
            self._tree.heading(ccol, text=ccol)
            self._tree.column(ccol, width=wdt, minwidth=40, anchor="center", stretch=True)
        self._tree.tag_configure("win", foreground=GREEN)
        self._tree.tag_configure("loss", foreground=RED)
        sc = ttk.Scrollbar(tf, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sc.set)
        self._tree.pack(side="left", fill="both", expand=True)
        sc.pack(side="left", fill="y")

        tk.Label(right, text="LIVE LOG", bg=DARK, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(8, 0))
        self._log_box = scrolledtext.ScrolledText(right, bg=PANEL, fg=TEXT, font=("Consolas", 9), width=40,
                                                  relief="flat", state="disabled", wrap="word")
        self._log_box.pack(fill="both", expand=True)
        for tag, col in (("info", TEXT), ("win", GREEN), ("loss", RED), ("warn", WARN),
                         ("error", RED), ("muted", MUTED), ("signal", GOLD)):
            self._log_box.tag_config(tag, foreground=col)

    # ─── config <-> fields ───
    _FIELD_MAP = [
        ("api_token", "_v_token"), ("app_id", "_v_app_id"), ("account_id", "_v_account_id"),
        ("symbol", "_v_symbol"), ("contract_mode", "_v_mode"), ("duration_min", "_v_duration"),
        ("multiplier", "_v_mult"), ("max_hold_min", "_v_hold"), ("min_payout_ratio", "_v_min_payout"),
        ("use_timesfm", "_v_use_tfm"), ("model_version", "_v_model"), ("device", "_v_device"),
        ("ctx_m5", "_v_ctx5"), ("ctx_m15", "_v_ctx15"), ("ctx_h1", "_v_ctx1h"), ("horizon_bars", "_v_hzn"),
        ("use_s1_band", "_v_s1"), ("use_s2_macro", "_v_s2"), ("use_s3_events", "_v_s3"),
        ("use_s4_mtf", "_v_s4"), ("use_s5_sizing", "_v_s5"), ("require_mtf_agreement", "_v_req_mtf"),
        ("w_prob", "_v_w_prob"), ("w_band", "_v_w_band"), ("w_macro", "_v_w_macro"), ("w_mtf", "_v_w_mtf"),
        ("min_confidence", "_v_conf"), ("min_prob", "_v_min_prob"),
        ("session_start_utc", "_v_sess_a"), ("session_end_utc", "_v_sess_b"), ("cooldown_bars", "_v_cool"),
        ("event_pre_min", "_v_pre"), ("event_post_min", "_v_post"),
        ("risk_pct", "_v_risk"), ("base_stake", "_v_stake"), ("min_stake", "_v_min_stake"),
        ("max_stake", "_v_max_stake"), ("use_martingale", "_v_use_mart"),
        ("martingale_multiplier", "_v_mart_mul"), ("max_martingale_steps", "_v_mart_steps"),
        ("max_daily_loss", "_v_max_loss"), ("max_daily_profit", "_v_max_profit"),
        ("max_consecutive_losses", "_v_max_consec"), ("max_trades_per_day", "_v_max_tpd"),
        ("max_retries", "_v_retries"),
        ("bt_bars", "_v_bt_bars"), ("bt_step", "_v_bt_step"), ("bt_payout", "_v_bt_payout"),
        ("bt_start_balance", "_v_bt_bal"),
    ]

    def _populate_fields(self):
        c = self.config_obj
        self._v_account_type.set("real" if c.use_real_account else "demo")
        for key, var in self._FIELD_MAP:
            getattr(self, var).set(getattr(c, key))
        self._events_box.delete("1.0", "end")
        self._events_box.insert("1.0", c.events_text)

    def _collect_config(self) -> GoldBotConfig:
        types = {k: type(v) for k, v in asdict(GoldBotConfig()).items()}
        kw = {}
        for key, var in self._FIELD_MAP:
            raw = getattr(self, var).get()
            t = types[key]
            kw[key] = raw.strip() if t is str else (bool(raw) if t is bool else t(float(raw)) if t is int else t(raw))
        kw["use_real_account"] = self._v_account_type.get() == "real"
        kw["events_text"] = self._events_box.get("1.0", "end").rstrip() + "\n"
        return GoldBotConfig(**kw)

    def _on_account_switch(self):
        try:
            out = self._collect_config()
            out.use_real_account = self.config_obj.use_real_account
            if self.config_obj is self._demo_config:
                self._demo_config = out
            else:
                self._real_config = out
        except Exception:
            pass
        if self._v_account_type.get() == "real":
            if not messagebox.askyesno("⚠ Real Account", "Switch to the REAL account settings?"):
                self._v_account_type.set("demo")
                return
        self.config_obj = self._real_config if self._v_account_type.get() == "real" else self._demo_config
        self.config_obj.use_real_account = self._v_account_type.get() == "real"
        self._populate_fields()

    # ─── actions ───
    def _save_config(self):
        try:
            cfg = self._collect_config()
            path = "gold_config_real.json" if cfg.use_real_account else "gold_config_demo.json"
            cfg.save(path)
            if cfg.use_real_account:
                self._real_config = cfg
            else:
                self._demo_config = cfg
            self.config_obj = cfg
            self._append_log(f"Config saved → {path}", "info")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _launch(self, mode: str):
        if not WEBSOCKETS_AVAILABLE or not NUMPY_AVAILABLE:
            messagebox.showerror("Missing dependency", "Run:  pip install websockets numpy")
            return
        try:
            cfg = self._collect_config()
        except Exception as e:
            messagebox.showerror("Config Error", str(e))
            return
        if not cfg.api_token or not cfg.account_id:
            messagebox.showerror("Missing credentials", "Enter your Deriv Access Token and Account ID.")
            return
        if mode == "live" and cfg.use_real_account:
            if not messagebox.askyesno("⚠ Real money",
                                       "You are about to trade with REAL MONEY.\n\n"
                                       "Have you backtested and forward-tested this on demo?"):
                return
        self.config_obj = cfg
        self._mode = mode
        self._reset_dashboard()
        self._engine = TradingEngine(cfg, self.msg_queue, mode=mode)
        threading.Thread(target=self._engine.run_in_thread, daemon=True).start()
        self._start_btn.state(["disabled"])
        self._bt_btn.state(["disabled"])
        self._stop_btn.state(["!disabled"])
        self._set_status("CONNECTING", WARN)
        self._append_log("Backtest starting…" if mode == "backtest" else "Bot starting…", "info")

    def _start_bot(self):
        self._launch("live")

    def _start_backtest(self):
        self._launch("backtest")

    def _stop_bot(self):
        if self._engine:
            self._engine.stop()
        self._start_btn.state(["!disabled"])
        self._bt_btn.state(["!disabled"])
        self._stop_btn.state(["disabled"])
        self._set_status("OFFLINE", MUTED)

    # ─── queue ───
    def _poll_queue(self):
        try:
            for _ in range(500):
                msg = self.msg_queue.get_nowait()
                try:
                    self._handle(msg)
                except Exception as e:
                    print(f"[UI ERROR] {msg.get('kind')}: {e}")
        except queue.Empty:
            pass
        self.after(40, self._poll_queue)

    def _handle(self, m: dict):
        k = m.get("kind")
        if k == "log":
            self._append_log(m["msg"], m.get("level", "info"))
        elif k == "status":
            self._set_status(m["text"], m["color"])
        elif k == "balance":
            self._balance_var.set(f"{m['value']:,.2f}")
        elif k == "model":
            self._model_var.set(f"model: {m['name']}")
        elif k == "tick":
            self._price_var.set(f"{m['price']:.2f}")
        elif k == "open_pnl":
            v = m["value"]
            self._stat_vars["open_pnl"].set(f"{v:+.2f}")
            self._stat_lbls["open_pnl"].config(fg=GREEN if v >= 0 else RED)
        elif k == "decision":
            d: Decision = m["dec"]
            if self._mode == "live":
                self._price_var.set(f"{d.price:.2f}")
            self._fchart.set(d.chart)
            blocked = next((r.replace("S3 event blackout: ", "") for r in d.reasons
                            if r.startswith("S3 event blackout")), "")
            self._bars.set(d.components, d.score, blocked)
            dv, dl = self._sig_vars["Direction"]
            dv.set(d.direction)
            dl.config(fg=GREEN if d.direction == "RISE" else RED if d.direction == "FALL" else MUTED)
            self._sig_vars["Confidence"][0].set(f"{d.confidence:.1f}%")
            self._sig_vars["P(up)"][0].set(f"{d.prob_up:.2f}")
            self._sig_vars["Regime"][0].set(d.regime)
        elif k == "signal":
            self._sig_vars["Stake"][0].set(f"{m['stake']:.2f}")
        elif k == "result":
            self._balance_var.set(f"{m['balance']:,.2f}")
            self._spark.add(m["daily_pnl"] if self._mode == "live" else
                            m["balance"] - self.config_obj.bt_start_balance)
            pnl = m["daily_pnl"]
            self._stat_vars["daily_pnl"].set(f"{pnl:+.2f}")
            self._stat_lbls["daily_pnl"].config(fg=GREEN if pnl >= 0 else RED)
            wr = m["win_rate"]
            self._stat_vars["win_rate"].set(f"{wr:.1f}%")
            self._stat_lbls["win_rate"].config(fg=GREEN if wr >= 55 else WARN if wr >= 50 else RED)
            self._stat_vars["total_trades"].set(str(m["total_trades"]))
            self._stat_vars["wins"].set(str(m["wins"]))
            self._stat_vars["losses"].set(str(m["losses"]))
            self._stat_vars["open_pnl"].set("—")
            when = m.get("when") or datetime.now().strftime("%H:%M:%S")
            self._tree.insert("", 0, values=(when, m["direction"], f"{m['stake']:.2f}",
                                             f"{m['profit']:+.2f}", f"{m['balance']:,.2f}"),
                              tags=("win" if m["is_won"] else "loss",))
            rows = self._tree.get_children()
            if len(rows) > 500:
                self._tree.delete(rows[-1])
        elif k == "progress":
            v, t = m["value"], m["total"]
            pct = int(v / t * 100)
            self._prog_var.set(pct)
            self._prog_lbl.config(text=f"Backtest {v}/{t} bars ({pct}%) — ETA {m.get('eta', 0):.0f}s")
        elif k == "stopped":
            self._start_btn.state(["!disabled"])
            self._bt_btn.state(["!disabled"])
            self._stop_btn.state(["disabled"])
            if self._mode == "backtest":
                self._prog_var.set(100)
            r = m.get("reason", "")
            if r and r not in ("Normal exit", "Engine loop exited"):
                self._append_log(f"⛔ Stopped: {r}", "error")
                self._set_status("OFFLINE", MUTED)

    def _reset_dashboard(self):
        for k, d in (("daily_pnl", "0.00"), ("win_rate", "0.0%"), ("total_trades", "0"),
                     ("wins", "0"), ("losses", "0"), ("open_pnl", "—")):
            self._stat_vars[k].set(d)
        for v, _ in self._sig_vars.values():
            v.set("—")
        self._spark.clear()
        self._fchart.set(None)
        self._bars.set({}, 0.0)
        self._prog_var.set(0)
        self._prog_lbl.config(text="")
        for r in self._tree.get_children():
            self._tree.delete(r)
        self._log_box.config(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.config(state="disabled")

    def _set_status(self, text, color):
        self._status_var.set(f"  {text}  ")
        self._status_lbl.config(bg=color, fg=DARK if color not in (MUTED, RED) else "white")

    def _append_log(self, msg, level="info"):
        self._log_box.config(state="normal")
        self._log_box.insert("end", msg + "\n", level)
        lines = int(self._log_box.index("end-1c").split(".")[0])
        if lines > 3000:
            self._log_box.delete("1.0", f"{lines - 3000}.0")
        self._log_box.see("end")
        self._log_box.config(state="disabled")


def main():
    if sys.platform == "win32":
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    GoldBotApp().mainloop()


if __name__ == "__main__":
    main()
