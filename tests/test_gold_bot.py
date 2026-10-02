"""
Unit tests for the TimesFM Gold Bot core (no network, no GUI).

    venv\\Scripts\\python -m unittest discover -s tests -v

Set RUN_MODEL_TESTS=1 to also load the real TimesFM model and run its self-test.
"""
import asyncio
import os
import queue
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gold_bot_gui as gb  # noqa: E402


def make_candles(closes, start=1_700_000_000, gran=300, opens=None):
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    epoch = start + gran * np.arange(n, dtype=np.int64)
    opens = np.asarray(opens, dtype=float) if opens is not None else np.r_[closes[0], closes[:-1]]
    return gb.Candles(epoch, opens, np.maximum(opens, closes) + 0.1,
                      np.minimum(opens, closes) - 0.1, closes)


class FakeForecaster:
    """Stats forecaster that records every call (to check caching/batching)."""

    def __init__(self, kind="stats", fail=False):
        self.kind = kind
        self.fail = fail
        self.calls = []

    @property
    def supports_covariates(self):
        return self.kind == "3.0"

    def forecast(self, contexts, horizon, pf_covs=None):
        contexts = [np.atleast_2d(np.asarray(c, dtype=np.float32)) for c in contexts]
        self.calls.append(([c.shape for c in contexts], pf_covs))
        out = [gb.Forecaster._fc_stats(c, horizon) for c in contexts]
        return out, ("stats" if self.fail else self.kind)


def base_cfg(**kw):
    cfg = gb.GoldBotConfig(use_s2_macro=False, use_s3_events=False, events_text="")
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# A Tuesday 10:00 UTC, inside the default 07–20 session
SESSION_NOW = 1_700_000_000 - (1_700_000_000 % 86400) + 86400 * 5 + 10 * 3600


class TestCovariateAlignment(unittest.TestCase):
    def test_prev_context_future_starts_at_the_next_bar(self):
        events = [(1_700_000_000 + 300 * 260, "NFP")]
        epochs = 1_700_000_000 + 300 * np.arange(256, dtype=np.int64)
        H = 12
        full = gb.pf_covariates(epochs, H, events)
        prev = gb.pf_covariates(epochs[:-1], H, events)
        self.assertEqual(full.shape, (2, 256 + H))
        self.assertEqual(prev.shape, (2, 255 + H))
        # the shorter context's "future" begins at the bar the full context ends on
        expected = gb.event_covariates(np.r_[epochs, epochs[-1] + 300 * np.arange(1, H)], events)
        np.testing.assert_allclose(prev, expected)
        np.testing.assert_allclose(prev, full[:, :255 + H])

    def test_future_columns_use_future_epochs(self):
        events = [(1_700_000_000 + 300 * 105, "CPI")]
        epochs = 1_700_000_000 + 300 * np.arange(100, dtype=np.int64)
        pf = gb.pf_covariates(epochs, 10, events)
        # proximity peaks exactly at the event bar (index 105 = context 100 + future step 5)
        self.assertEqual(int(np.argmax(pf[0])), 105)


class TestProbAbove(unittest.TestCase):
    q = np.array([90, 92, 94, 96, 100, 104, 106, 108, 110], dtype=float)

    def test_median_is_half(self):
        self.assertAlmostEqual(gb.prob_above(100, self.q), 0.5)

    def test_monotone_and_clamped(self):
        levels = np.linspace(60, 140, 200)
        ps = [gb.prob_above(x, self.q) for x in levels]
        self.assertTrue(all(a >= b - 1e-12 for a, b in zip(ps, ps[1:])))
        self.assertLessEqual(max(ps), 0.98)
        self.assertGreaterEqual(min(ps), 0.02)


class TestHelpers(unittest.TestCase):
    def test_closed_until_has_no_lookahead(self):
        c = make_candles(np.arange(10) + 100.0)
        now = c.epoch[5] + 299           # bar 5 still forming
        self.assertEqual(len(c.closed_until(now, 300)), 5)
        self.assertEqual(len(c.closed_until(c.epoch[5] + 300, 300)), 6)

    def test_wilson(self):
        lo, hi = gb.wilson_interval(55, 100)
        self.assertLess(lo, 0.55)
        self.assertGreater(hi, 0.55)
        self.assertLess(lo, 0.5)          # 55/100 does not rule out a coin flip
        lo, _ = gb.wilson_interval(600, 1000)
        self.assertGreater(lo, 0.56)

    def test_utc_dt(self):
        self.assertEqual(gb.utc_dt(0).year, 1970)
        self.assertIsNotNone(gb.utc_dt(0).tzinfo)


class TestForecasterValidation(unittest.TestCase):
    def test_stats_shapes(self):
        ctx = np.vstack([np.linspace(2600, 2610, 300), np.linspace(100, 101, 300)])
        med, q = gb.Forecaster._fc_stats(ctx, 7)
        self.assertEqual(med.shape, (2, 7))
        self.assertEqual(q.shape, (2, 7, 9))
        self.assertTrue(np.all(np.diff(q, axis=-1) >= 0))

    def test_validate_rejects_bad_shapes(self):
        ctx = np.ones((1, 50))
        good = [(np.ones((1, 4)), np.ones((1, 4, 9)))]
        gb.Forecaster._validate(good, [ctx], 4)
        for bad in ([(np.ones((1, 4)), np.ones((1, 4, 10)))],     # 10 quantile columns
                    [(np.ones((1, 3)), np.ones((1, 3, 9)))],      # wrong horizon
                    [(np.full((1, 4), np.nan), np.ones((1, 4, 9)))]):
            with self.assertRaises(ValueError):
                gb.Forecaster._validate(bad, [ctx], 4)

    def test_broken_model_falls_back_and_reports_source(self):
        f = gb.Forecaster(base_cfg(), log=lambda *a: None)
        f.kind = "3.0"
        f.model = mock.Mock()
        f.model.predict_batch.side_effect = RuntimeError("boom")
        out, src = f.forecast([np.linspace(1, 2, 100)], 5)
        self.assertEqual(src, "stats")
        self.assertEqual(f.fallbacks, 1)
        self.assertEqual(out[0][1].shape, (1, 5, 9))

    def test_v25_rejects_unexpected_layout(self):
        f = gb.Forecaster(base_cfg(), log=lambda *a: None)
        f.kind = "2.5"
        f.model = mock.Mock()
        f.model.forecast.return_value = (np.ones((1, 5)), np.ones((1, 5, 9)))   # missing mean col
        _, src = f.forecast([np.linspace(1, 2, 100)], 5)
        self.assertEqual(src, "stats")


class TestStrategy(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.m5 = make_candles(2600 + np.cumsum(rng.normal(0, 0.5, 600)),
                               start=SESSION_NOW - 600 * 300)
        self.m15 = make_candles(2600 + np.cumsum(rng.normal(0, 0.8, 200)),
                                start=SESSION_NOW - 200 * 900, gran=900)
        self.h1 = make_candles(2600 + np.cumsum(rng.normal(0, 1.5, 200)),
                               start=SESSION_NOW - 200 * 3600, gran=3600)

    def test_missing_mtf_data_vetoes(self):
        s = gb.GoldStrategy(base_cfg(min_confidence=0, min_prob=0), FakeForecaster())
        d = s.evaluate(SESSION_NOW, self.m5, None, None, {}, [])
        self.assertEqual(d.direction, "FLAT")
        self.assertTrue(any("S4 M15/H1 data unavailable" in r for r in d.reasons))

    def test_model_failure_vetoes(self):
        s = gb.GoldStrategy(base_cfg(min_confidence=0, min_prob=0, use_s4_mtf=False),
                            FakeForecaster(kind="3.0", fail=True))
        d = s.evaluate(SESSION_NOW, self.m5, None, None, {}, [])
        self.assertEqual(d.direction, "FLAT")
        self.assertTrue(any("stats fallback" in r for r in d.reasons))

    def test_prev_bar_forecast_is_cached(self):
        fc = FakeForecaster()
        s = gb.GoldStrategy(base_cfg(use_s4_mtf=False), fc)
        s.evaluate(SESSION_NOW, self.m5.slice(0, 599), None, None, {}, [])
        self.assertEqual(len(fc.calls[-1][0]), 2)        # cold: current + previous-bar context
        s.evaluate(SESSION_NOW + 300, self.m5, None, None, {}, [])
        self.assertEqual(len(fc.calls[-1][0]), 1)        # warm: previous bar reused

    def test_cached_prev_forecast_matches_recomputed(self):
        a = gb.GoldStrategy(base_cfg(use_s4_mtf=False), FakeForecaster())
        a.evaluate(SESSION_NOW, self.m5.slice(0, 599), None, None, {}, [])
        da = a.evaluate(SESSION_NOW + 300, self.m5, None, None, {}, [])
        b = gb.GoldStrategy(base_cfg(use_s4_mtf=False), FakeForecaster())
        b.width_hist.extend(a.width_hist)
        b.width_hist.pop()
        db = b.evaluate(SESSION_NOW + 300, self.m5, None, None, {}, [])
        self.assertAlmostEqual(da.components["S1 Band"], db.components["S1 Band"], places=5)

    def test_mtf_forecast_reused_until_new_bar(self):
        fc = FakeForecaster()
        s = gb.GoldStrategy(base_cfg(), fc)
        s.evaluate(SESSION_NOW, self.m5, self.m15, self.h1, {}, [])
        n = len(fc.calls)
        s.evaluate(SESSION_NOW, self.m5, self.m15, self.h1, {}, [])
        mtf_calls = [c for c in fc.calls[n:] if c[0] == [(1, 200), (1, 200)]]
        self.assertEqual(mtf_calls, [])

    def test_correlated_signal_is_down_weighted(self):
        s = gb.GoldStrategy(base_cfg(), FakeForecaster())
        rng = np.random.default_rng(1)
        for _ in range(100):
            p = rng.normal()
            s._comp_hist.append({"Prob": p, "S1 Band": p + rng.normal(0, 0.05),
                                 "S2 Macro": rng.normal()})
        self.assertGreater(s._corr_with_prob("S1 Band"), 0.95)
        self.assertLess(abs(s._corr_with_prob("S2 Macro")), 0.3)


class TestSecrets(unittest.TestCase):
    def test_roundtrip(self):
        enc = gb.protect_secret("tok-123")
        self.assertNotIn("tok-123", enc)
        self.assertEqual(gb.unprotect_secret(enc), "tok-123")
        if sys.platform == "win32":
            self.assertTrue(enc.startswith("dpapi:"))

    def test_config_never_stores_plaintext_and_migrates_legacy(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "cfg.json")
            with open(p, "w") as f:
                f.write('{"api_token": "legacy-secret", "account_id": "X1"}')
            cfg = gb.GoldBotConfig.load(p)
            self.assertEqual(cfg.api_token, "legacy-secret")
            with open(p) as f:
                self.assertNotIn("legacy-secret", f.read())        # migrated on load
            self.assertEqual(gb.GoldBotConfig.load(p).api_token, "legacy-secret")


class TestBacktestFills(unittest.TestCase):
    def _run(self, closes, opens, spread, payout_quote=None):
        cfg = base_cfg(bt_bars=40, bt_step=1, ctx_m5=64, duration_min=5, cooldown_bars=0,
                       bt_spread=spread, bt_payout=0.85, use_s5_sizing=False, base_stake=10,
                       max_trades_per_day=1000, max_consecutive_losses=1000,
                       max_daily_loss=1e9, max_daily_profit=1e9)
        eng = gb.TradingEngine(cfg, queue.Queue(), mode="backtest")
        eng._log = lambda *a, **k: None
        m5 = make_candles(closes, start=SESSION_NOW - len(closes) * 300 + 3600, opens=opens)

        async def fetch_all(*_):
            return m5, None, None, {}

        async def quote():
            return payout_quote

        eng._fetch_all = fetch_all
        eng._quote_payout_ratio = quote
        eng.strategy.evaluate = lambda *a, **k: gb.Decision(direction="RISE", model_src="stats")
        eng.loop = asyncio.new_event_loop()
        try:
            eng.loop.run_until_complete(eng._backtest())
        finally:
            eng.loop.close()
        return eng

    def test_entry_is_next_open_plus_half_spread(self):
        n = 120
        closes = np.full(n, 100.0) + 0.1 * (np.arange(n) % 2)   # close alternates 100.0/100.1
        opens = closes - 0.05
        # close-to-close every other trade would win; with entry at next open +0.2 none can
        eng = self._run(closes, opens, spread=0.4)
        self.assertGreater(eng.total_trades, 0)
        self.assertEqual(eng.wins, 0)

    def test_without_spread_next_open_entry_wins(self):
        n = 120
        closes = np.full(n, 100.0)
        opens = closes - 0.05                                     # every bar closes above its open
        eng = self._run(closes, opens, spread=0.0)
        self.assertGreater(eng.total_trades, 0)
        self.assertEqual(eng.losses, 0)

    def test_quoted_payout_caps_config(self):
        n = 120
        eng = self._run(np.full(n, 100.0), np.full(n, 99.95), spread=0.0, payout_quote=0.70)
        self.assertGreater(eng.total_trades, 0)
        self.assertAlmostEqual(eng.daily_pnl / eng.total_trades, 7.0, places=2)   # 10 × 0.70


@unittest.skipUnless(os.environ.get("RUN_MODEL_TESTS") == "1", "set RUN_MODEL_TESTS=1")
class TestRealModel(unittest.TestCase):
    def test_load_and_self_test(self):
        f = gb.Forecaster(base_cfg(), log=lambda m, *a: gb.safe_print(m))
        f.load()
        self.assertIn(f.kind, ("3.0", "2.5"))
        out, src = f.forecast([np.linspace(2600, 2610, 300)[None, :]], 6)
        self.assertEqual(src, f.kind)
        self.assertEqual(out[0][1].shape, (1, 6, 9))


if __name__ == "__main__":
    unittest.main()
