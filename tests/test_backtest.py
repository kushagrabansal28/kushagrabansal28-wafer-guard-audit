"""Tests for the rolling out-of-time backtest windows. Run:  python -m pytest -q"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt  # noqa: E402


def test_windows_are_strictly_time_ordered_and_tie_safe():
    ts = np.repeat(np.arange(300), 2).astype("datetime64[m]")          # every timestamp appears twice
    wins = bt.backtest_windows(ts)
    assert len(wins) == 5
    for itr, iva, ite in wins:
        assert ts[itr].max() < ts[iva].min() and ts[iva].max() < ts[ite].min()
        assert set(itr).isdisjoint(iva) and set(iva).isdisjoint(ite) and set(itr).isdisjoint(ite)


def test_each_window_trains_only_on_the_past_and_windows_do_not_overlap():
    ts = np.arange(1000).astype("datetime64[m]")
    wins = bt.backtest_windows(ts)
    tests = [w[2] for w in wins]
    for k, (itr, iva, ite) in enumerate(wins):
        assert max(itr.max(), iva.max()) < ite.min()                     # everything fitted on precedes the window
        if k:
            assert tests[k - 1].max() < tests[k].min()                   # consecutive, non-overlapping test windows
            assert tests[k - 1].max() < iva.max() + 1                    # the previous window is now part of the past
    assert tests[-1].max() == len(ts) - 1                                # the last window reaches the end


def test_empty_window_is_rejected():
    with pytest.raises(ValueError):
        bt.backtest_windows(np.arange(6).astype("datetime64[m]"), n_windows=5)
