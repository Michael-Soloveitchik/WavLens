"""Numerical verification of wavlens weighting / interpolation / function binding / rates.

    nice python3 tests/test_weights.py      # from the repo root
"""
import base64
import io
import os
import sys
import traceback

import numpy as np
import soundfile as sf
from scipy.interpolate import CubicSpline
from scipy.signal import resample_poly

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import wavlens  # noqa: E402
from pyannote.core import SlidingWindow, SlidingWindowFeature  # noqa: E402

TOL = 2e-4            # 16-bit WAV round-trip + float32 arithmetic
QTOL = 2.0 / 32768    # "exact up to quantization"
SR, DUR = 16000, 2.0
N = int(SR * DUR)
rng = np.random.default_rng(0)
t = np.arange(N) / SR
X = (0.6 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.standard_normal(N)).astype(np.float32)   # loud
Y = (0.2 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(N)).astype(np.float32)   # quiet
XA, YA = (X, SR), (Y, SR)   # every array audio carries its rate

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
        print(f"PASS  {name}")
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        if not isinstance(e, AssertionError):
            msg += "\n" + traceback.format_exc().strip().splitlines()[-3]
        RESULTS.append((name, False, msg))
        print(f"FAIL  {name}\n      {msg}")


def channels(w, sr=SR):
    assert w.audio_url.startswith("data:audio/wav;base64,"), w.audio_url[:40]
    raw = base64.b64decode(w.audio_url.split(",", 1)[1])
    data, got_sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    assert got_sr == sr, f"wav rate {got_sr} != {sr}"
    return data.T  # (C, T)


def ref_resample(w, n, method):
    """Independent reference for the documented convention: sample k of L sits at (k + 0.5) / L,
    output point i of n at (i + 0.5) / n. L > n: strongest |value| per bin of ceil(L / n) (sign kept).
    Index math for nearest / previous is exact-integer."""
    w = np.asarray(w, np.float64).reshape(-1)
    L = len(w)
    if L == n:
        return w
    if L > n:
        k = -(-L // n)
        b = np.pad(w, (0, (-L) % k), mode="edge").reshape(-1, k)
        return b[np.arange(len(b)), np.abs(b).argmax(1)]
    i = np.arange(n)
    if method == "previous":
        return w[np.minimum(((2 * i + 1) * L) // (2 * n), L - 1)]
    if method == "nearest":  # round((i+.5)/n*L - .5) = floor(((2i+1)L) / (2n)) away from exact ties
        return w[np.clip(((2 * i + 1) * L) // (2 * n), 0, L - 1)]
    src, dst = np.arange(L) + 0.5, (i + 0.5) / n * L
    if method == "linear":
        return np.interp(dst, src, w)
    if method == "cubic":
        return np.clip(CubicSpline(src, w)(dst), w.min(), w.max())
    raise ValueError(method)


def expected(audios, weights, method="linear"):
    """weighted channels / global peak, mirroring float32 arithmetic."""
    out = []
    for x, w in zip(audios, weights):
        w = np.asarray(w, np.float32)
        out.append(x * (np.float32(w) if w.ndim == 0 else ref_resample(w, len(x), method).astype(np.float32)))
    peak = max(float(np.abs(o).max()) for o in out) or 1.0
    return [o / peak for o in out]


def assert_close(got, exp, tol, what):
    assert got.shape == exp.shape, f"{what}: shape {got.shape} != {exp.shape}"
    err = float(np.abs(got - exp).max())
    assert err <= tol, f"{what}: max abs error {err:.3e} > {tol:.1e}"


def raises(exc, fn, what):
    try:
        fn()
    except exc:
        return
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"{what}: raised {type(e).__name__} instead of {exc.__name__}: {e}")
    raise AssertionError(f"{what}: did not raise {exc.__name__}")


# ---------------------------------------------------------------- 0. rates
def t0_bare_array_raises():
    raises(ValueError, lambda: wavlens.show(X), "positional bare array")
    raises(ValueError, lambda: wavlens.show(audio=X), "keyword bare array")
    raises(ValueError, lambda: wavlens.show(audio={"a": XA, "b": Y}), "bare array inside dict")
    raises(ValueError, lambda: wavlens.show(audio=[XA, Y]), "bare array inside list")
    raises(ValueError, lambda: wavlens.show(X, sample_rate=SR), "bare array even with a target rate")


def t0_rate_forms():
    a = channels(wavlens.show(XA))
    b = channels(wavlens.show({"waveform": X, "sample_rate": SR}))
    assert a.shape == b.shape == (1, N), (a.shape, b.shape)
    assert_close(a[0], b[0], 0.0, "tuple vs waveform dict")
    assert_close(a[0], X / float(np.abs(X).max()), TOL, "tuple form")


def t0_target_rate():
    w = wavlens.show(XA, sample_rate=8000)
    ch = channels(w, sr=8000)
    assert ch.shape == (1, N // 2), ch.shape
    assert abs(w.duration - DUR) < 1e-6, w.duration
    ref = resample_poly(X, 1, 2).astype(np.float32)
    assert_close(ch[0], ref / float(np.abs(ref).max()), TOL, "target 8 kHz")


def t0_mixed_rates_highest():
    y8 = resample_poly(Y, 1, 2).astype(np.float32)          # 8 kHz copy of Y
    w = wavlens.show(audio={"a": XA, "b": (y8, 8000)})       # no target: highest rate wins
    ch = channels(w, sr=SR)
    assert ch.shape == (2, N), ch.shape
    assert abs(w.duration - DUR) < 1e-6, w.duration
    up = resample_poly(y8, 2, 1).astype(np.float32)
    exp = expected([X, up], [1.0, 1.0])
    assert_close(ch[0], exp[0], TOL, "mixed rates ch0 (16 k untouched)")
    assert_close(ch[1], exp[1], TOL, "mixed rates ch1 (8 k -> 16 k)")


# ---------------------------------------------------------------- 1. scalar weight
def t1_scalar_single():
    w = wavlens.show(XA, weights=0.3)
    ch = channels(w)
    exp = expected([X], [0.3])
    assert ch.shape[0] == 1 and w.weighted is True
    assert_close(ch[0], exp[0], TOL, "scalar single")


def t1_scalar_list():
    w = wavlens.show(audio=[XA, YA], weights=[0.3, 1.0])
    ch = channels(w)
    exp = expected([X, Y], [0.3, 1.0])
    assert ch.shape[0] == 2
    assert_close(ch[0], exp[0], TOL, "scalar list ch0")
    assert_close(ch[1], exp[1], TOL, "scalar list ch1")


# ---------------------------------------------------------------- 2. elementwise weight
def t2_elementwise():
    ramp = np.linspace(0, 1, N, dtype=np.float32)
    w = wavlens.show(audio={"a": XA, "b": YA}, weights={"a": ramp, "b": 1.0})
    ch = channels(w)
    exp = expected([X, Y], [ramp, 1.0])
    assert_close(ch[0], exp[0], QTOL, "elementwise ch0")
    assert_close(ch[1], exp[1], QTOL, "elementwise ch1")


# ---------------------------------------------------------------- 3. interpolated weights
def make_t3(m, method):
    def fn():
        wv = rng.uniform(0.2, 1.0, m).astype(np.float32)
        w = wavlens.show(audio={"a": XA, "b": YA}, weights={"a": wv, "b": 0.7}, interpolation=method)
        ch = channels(w)
        exp = expected([X, Y], [wv, 0.7], method)
        assert_close(ch[0], exp[0], TOL, f"len {m} {method} ch0")
        assert_close(ch[1], exp[1], TOL, f"len {m} {method} ch1")
    return fn


def t3_pooled():
    wv = rng.uniform(-1.0, 1.0, 3 * N).astype(np.float32)   # longer than the audio: max-|.| pooled
    w = wavlens.show(audio={"a": XA, "b": YA}, weights={"a": wv, "b": 0.7})
    ch = channels(w)
    exp = expected([X, Y], [wv, 0.7])
    assert_close(ch[0], exp[0], TOL, "pooled weight ch0")
    assert_close(ch[1], exp[1], TOL, "pooled weight ch1")


# ---------------------------------------------------------------- 4. forms
def t4_dict_forms():
    wavlens.show(audio={"a": XA, "b": YA}, weights={"a": 0.5, "b": 0.2})
    raises(ValueError, lambda: wavlens.show(audio={"a": XA, "b": YA}, weights={"a": 0.5, "c": 0.2}), "dict wrong keys")
    raises(ValueError, lambda: wavlens.show(audio={"a": XA, "b": YA}, weights={"a": 0.5}), "dict missing key")
    raises(ValueError, lambda: wavlens.show(audio={"a": XA, "b": YA}, weights=[0.5, 0.2]), "dict audio, list weights")
    raises(ValueError, lambda: wavlens.show(audio={"a": XA, "b": YA}, weights=0.5), "dict audio, scalar weights")


def t4_list_forms():
    wavlens.show(audio=[XA, YA], weights=[1, 0.5])
    raises(ValueError, lambda: wavlens.show(audio=[XA, YA], weights={"a": 1, "b": 0.5}), "list audio, dict weights")
    raises(ValueError, lambda: wavlens.show(audio=[XA, YA], weights=[1, 0.5, 0.2]), "list audio, wrong length")
    raises(ValueError, lambda: wavlens.show(audio=[XA, YA], weights=[1]), "list audio, too short")
    raises(ValueError, lambda: wavlens.show(audio=[XA, YA], weights=0.5), "list audio, scalar weights")


def t4_single_forms():
    a = wavlens.show(audio=XA, weights=0.3)
    b = wavlens.show(audio=XA, weights=np.linspace(0, 1, N))
    c = wavlens.show(audio=XA, weights=np.array([0.2, 1.0, 0.5]))
    assert all(v.weighted for v in (a, b, c))
    assert_close(channels(c)[0], expected([X], [np.array([0.2, 1.0, 0.5])])[0], TOL, "single 3-pt weight")


def t4_positional():
    a = wavlens.show(XA, weights=0.3)
    assert a.channels == 1 and a.weighted
    assert_close(channels(a)[0], expected([X], [0.3])[0], TOL, "positional single")
    b = wavlens.show(XA, YA, weights=[1, 0.5])
    assert b.channels == 2 and b.audio_names == ["Audio_0", "Audio_1"], b.audio_names
    ch, exp = channels(b), expected([X, Y], [1, 0.5])
    assert_close(ch[0], exp[0], TOL, "positional list ch0")
    assert_close(ch[1], exp[1], TOL, "positional list ch1")
    raises(ValueError, lambda: wavlens.show(XA, YA, weights=0.5), "positional two audios, scalar weights")


# ---------------------------------------------------------------- 5. default
def t5_default():
    w = wavlens.show(audio={"a": XA, "b": YA}, sample_rate=SR)   # target equal to the carried rate: no-op
    assert w.weighted is False, w.weighted
    ch = channels(w)
    exp = expected([X, Y], [1.0, 1.0])
    assert_close(ch[0], exp[0], TOL, "default ch0")
    assert_close(ch[1], exp[1], TOL, "default ch1")
    assert len(w.peaks) == 3 and all(len(p) == 2000 for p in w.peaks), [len(p) for p in w.peaks]  # 2 audios + the mix row


# ---------------------------------------------------------------- 6. relative levels
def t6_relative_levels():
    wa, wb = 0.5, 2.0
    w = wavlens.show(audio={"a": XA, "b": YA}, weights={"a": wa, "b": wb})
    ch = channels(w)
    rms = lambda v: float(np.sqrt(np.mean(np.square(v, dtype=np.float64))))
    got = rms(ch[0]) / rms(ch[1])
    want = (wa * rms(X)) / (wb * rms(Y))
    assert abs(got - want) / want < 1e-3, f"RMS ratio {got:.5f} vs weighted {want:.5f}"


# ---------------------------------------------------------------- 7. functions
F1, F2, F3 = rng.standard_normal(2000), rng.standard_normal(500), rng.standard_normal(N)
F4, F5 = rng.standard_normal(10), rng.standard_normal(33)
X0, X1 = 0.5 / 2000, 1 - 0.5 / 2000   # frame-centre convention


def _check_fn(f, name, row):
    assert f["name"] == name, f"name {f['name']!r} != {name!r}"
    assert f["row"] == row, f"{name}: row {f['row']} != {row}"
    assert len(f["x"]) == 2000 and len(f["y"]) == 2000, f"{name}: {len(f['x'])} x, {len(f['y'])} y"
    assert min(f["y"]) == 0.0 and max(f["y"]) == 1.0, f"{name}: y range {min(f['y'])}..{max(f['y'])}"
    assert abs(f["x"][0] - X0) < 1e-5 and abs(f["x"][-1] - X1) < 1e-5, f"{name}: x span {f['x'][0]}..{f['x'][-1]}"


def t7_binding():
    w = wavlens.show(audio={"a": XA, "b": YA}, function={"a": F1, "b": [F2, F3], "glob": F5})
    fs = w.functions
    assert [f["name"] for f in fs] == ["a", "b[0]", "b[1]", "glob"], [f["name"] for f in fs]
    for f, (nm, row) in zip(fs, [("a", 0), ("b[0]", 1), ("b[1]", 1), ("glob", -1)]):
        _check_fn(f, nm, row)


def t7_nested_dict_non_audio_key():
    w = wavlens.show(audio={"a": XA, "b": YA}, function={"c": {"u": F4}})
    fs = w.functions
    assert len(fs) == 1, [f["name"] for f in fs]
    assert fs[0]["name"] in ("c.u", "u", "c"), fs[0]["name"]
    _check_fn(fs[0], fs[0]["name"], -1)


def t7_previous_stepwise():
    f = rng.standard_normal(37)
    w = wavlens.show(audio=XA, function={"f": f}, interpolation="previous")
    y = np.array(w.functions[0]["y"])
    assert len(y) == 2000
    nd = len(np.unique(y))
    assert nd <= 37, f"previous: {nd} distinct values > 37"
    ref = ref_resample(f, 2000, "previous")
    ref = (ref - ref.min()) / (ref.max() - ref.min())
    assert_close(y, ref, 6e-4, "previous function y")  # y rounded to 3 decimals


def t7_sliding_window_feature():
    sw = SlidingWindow(start=0.0, duration=0.05, step=0.02)
    nfr = 100
    swf = SlidingWindowFeature(rng.standard_normal((nfr, 1)).astype(np.float32), sw)
    w = wavlens.show(audio=XA, function={"swf": swf})
    f = w.functions[0]
    assert f["name"] == "swf" and f["row"] == -1, (f["name"], f["row"])
    want = (0.025 + 0.02 * np.arange(nfr)) / DUR
    want = want[want <= 1.0]   # frames past the audio are dropped
    assert len(f["x"]) == len(want), (len(f["x"]), len(want))
    assert_close(np.array(f["x"]), want, 1e-3, "swf x")
    assert min(f["y"]) == 0.0 and max(f["y"]) == 1.0
    swf2 = SlidingWindowFeature(rng.standard_normal((nfr, 2)).astype(np.float32), sw)
    names = [f["name"] for f in wavlens.show(audio=XA, function={"m": swf2}).functions]
    assert names == ["m[0]", "m[1]"], names


# ---------------------------------------------------------------- 8. _resample
def t8_resample_identity():
    y = rng.standard_normal(123).astype(np.float32)
    for m in ("linear", "nearest", "previous", "cubic"):
        r = wavlens._resample(y, 123, m)
        assert r.dtype == np.float32 and np.array_equal(r, y), m
    y64 = rng.standard_normal(50)
    assert np.array_equal(wavlens._resample(y64, 50, "linear"), y64.astype(np.float32))


def t8_bad_method():
    raises(ValueError, lambda: wavlens._resample(np.arange(5.0), 10, "spline"), "_resample spline")
    raises(ValueError, lambda: wavlens._resample(np.arange(5.0), 10, "quadratic"), "_resample quadratic")
    raises(ValueError, lambda: wavlens.show(audio=XA, weights=np.array([1.0, 0.5]), interpolation="spline"), "show spline")
    raises(ValueError, lambda: wavlens._weight(X, np.array([1.0, 0.5]), "spline"), "_weight spline")


if __name__ == "__main__":
    check("0a bare array raises ValueError", t0_bare_array_raises)
    check("0b (array, sr) == {'waveform', 'sample_rate'}", t0_rate_forms)
    check("0c sample_rate= target resamples to 8 kHz", t0_target_rate)
    check("0d mixed 8 k + 16 k without target -> 16 kHz", t0_mixed_rates_highest)
    check("1a scalar weight, single audio", t1_scalar_single)
    check("1b scalar weights, list audio", t1_scalar_list)
    check("2  elementwise weight (len = audio)", t2_elementwise)
    for m in (7, 1000, 2):
        for method in ("linear", "nearest", "previous", "cubic"):
            check(f"3  weight len {m:4d} {method}", make_t3(m, method))
    check("3  weight len 96000 pooled (max |.| per bin)", t3_pooled)
    check("4a dict forms", t4_dict_forms)
    check("4b list forms", t4_list_forms)
    check("4c single forms", t4_single_forms)
    check("4d positional forms", t4_positional)
    check("5  default weights=None (target == carried rate)", t5_default)
    check("6  relative levels preserved", t6_relative_levels)
    check("7a function binding rows/names/scaling", t7_binding)
    check("7b nested dict under non-audio key", t7_nested_dict_non_audio_key)
    check("7c previous interpolation is stepwise", t7_previous_stepwise)
    check("7d SlidingWindowFeature keeps its time axis", t7_sliding_window_feature)
    check("8a _resample equal length identity", t8_resample_identity)
    check("8b unknown interpolation raises", t8_bad_method)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    sys.exit(1 if n_fail else 0)
