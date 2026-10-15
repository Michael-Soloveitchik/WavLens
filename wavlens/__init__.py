"""wavlens: waveform + pyannote annotations + moving playhead, interactive in Jupyter.

    import wavlens
    wavlens.show("audio.wav", "audio.rttm")                     # positional, any order, objects or paths
    wavlens.show(audio={"clean": x, "noisy": y}, annotation={"ref": ref, "hyp": hyp},
               function={"vad": p}, sample_rate=16000)        # named rows
"""

import base64
import hashlib
import io
import math
import json
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anywidget
import numpy as np
import soundfile as sf
import traitlets
from pyannote.core import Annotation, Segment, SlidingWindowFeature, Timeline

__version__ = "0.1.0"
__all__ = ["show", "WavLens", "PALETTE"]

PALETTE = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999"]  # Set1, as pyannote
LINE_PALETTE = ["#000000", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
ANN_EXT = {".rttm", ".uem", ".lab", ".txt", ".json", ".seglst", ".jsonl"}
METHODS = ("linear", "nearest", "previous", "cubic")
N_POINTS = 2000          # peaks per audio and points per function sent to the browser
STATE_WARN_MB = 20       # widget state above this lands in the notebook file; warn


# ----------------------------------------------------------------------------- arrays

def _np(a, what="array"):
    """array-like / torch (any device, any dtype) -> finite float32 numpy. Integers are scaled to ±1."""
    if hasattr(a, "detach"):
        a = a.detach().cpu()
        a = a.float().numpy() if "float" in str(a.dtype) else a.numpy()
    a = np.asarray(a)
    if a.dtype.kind in "iu":
        a = a / np.iinfo(a.dtype).max
    elif a.dtype.kind == "b":
        a = a.astype(np.float32)
    elif a.dtype.kind != "f":
        raise TypeError(f"{what}: expected numeric data, got dtype {a.dtype}")
    a = a.astype(np.float32, copy=False)
    if a.size and not np.isfinite(a).all():
        raise ValueError(f"{what}: {int((~np.isfinite(a)).sum())} non-finite values (NaN/inf)")
    return a


def _is_array(o):
    return isinstance(o, np.ndarray) or hasattr(o, "detach")


def _is_ann_path(o):
    return isinstance(o, (str, Path)) and Path(o).suffix.lower() in ANN_EXT


def _is_ann_obj(o):  # DiarizeOutput, HF ModelOutput (.logits), {"diarization": [...]}, NeMo manifest dict, RTTM text
    return hasattr(o, "speaker_diarization") or hasattr(o, "logits") \
        or (isinstance(o, dict) and ("diarization" in o or "rttm_filepath" in o)) \
        or (isinstance(o, str) and o.lstrip().startswith("SPEAKER "))


def _is_num(t):
    try:
        float(t)
        return True
    except ValueError:
        return False


def _is_lines(o):   # NeMo Sortformer diarize(): ["0.00 4.98 speaker_0", ...]
    return isinstance(o, (list, tuple)) and len(o) > 0 and \
        all(isinstance(e, str) and len(f := e.split()) >= 2 and _is_num(f[0]) and _is_num(f[1]) for e in o)


def _is_records(o):  # [{start|start_time, end|end_time, speaker|label}, ...]
    return isinstance(o, (list, tuple)) and len(o) > 0 and all(isinstance(e, dict) and ("start" in e or "start_time" in e) for e in o)


def _is_tuples(o):   # [(start, end[, label]), ...]
    return isinstance(o, (list, tuple)) and len(o) > 0 and \
        all(isinstance(e, (list, tuple)) and len(e) in (2, 3) and isinstance(e[0], (int, float, np.number)) for e in o)


def _is_speaker_dict(o):  # {speaker: [(start, end), ...] | Timeline | Segment}: ONE annotation, labels = keys
    def bare(v):
        return isinstance(v, (Timeline, Segment)) or (isinstance(v, (list, tuple)) and len(v) > 0 and all(
            isinstance(e, Segment) or (isinstance(e, (list, tuple)) and len(e) == 2 and isinstance(e[0], (int, float, np.number)))
            for e in v))
    return isinstance(o, dict) and len(o) > 0 and all(bare(v) for v in o.values())


def _is_ann(o):
    return isinstance(o, (Annotation, Timeline, Segment)) or _is_ann_path(o) or _is_ann_obj(o) \
        or _is_lines(o) or _is_records(o) or _is_tuples(o) or _is_speaker_dict(o)


def _is_audio(o):
    return (isinstance(o, (str, Path)) and not _is_ann_path(o) and not _is_ann_obj(o)) or _is_array(o) \
        or (isinstance(o, dict) and ("audio" in o or "waveform" in o)) \
        or (isinstance(o, tuple) and len(o) == 2 and isinstance(o[1], (int, float, np.number)) and not isinstance(o[1], bool))


def _is_fn(o):
    return _is_array(o) or isinstance(o, SlidingWindowFeature) or hasattr(o, "logits")


def _named(obj, is_one):
    """single | list | {name: item} -> [(name, item), ...]; names are str."""
    if obj is None:
        return []
    if is_one(obj):
        return [(None, obj)]
    if isinstance(obj, dict):
        return [(str(k), v) for k, v in obj.items()]
    if isinstance(obj, (list, tuple)):
        return [(None, o) for o in obj]
    return [(None, obj)]


def _unique(names, prefix):
    """Fill None names with prefix / prefix_<i>, unique; explicit duplicates get ~1, ~2 (auto names yield)."""
    explicit = [str(n) for n in names if n is not None]
    seen, out = set(), []
    for i, n in enumerate(names):
        if n is None:
            n = prefix if len(names) == 1 else f"{prefix}_{i}"
            while n in explicit or n in seen:
                n = n + "_"
        else:
            base, k, n = str(n), 1, str(n)
            while n in seen:
                n, k = f"{base}~{k}", k + 1
        seen.add(n)
        out.append(n)
    return out


# ----------------------------------------------------------------------------- cache

def _nbytes(v):
    if isinstance(v, (str, bytes)):
        return len(v)
    if isinstance(v, (list, tuple)):
        return sum(map(_nbytes, v)) + 8 * len(v)
    return getattr(v, "nbytes", 8)


class _Cache(OrderedDict):
    """Tiny LRU keyed by content hash, capped by total bytes. Re-showing the same audio in another
    cell reuses its WAV, peaks and spectrogram instead of recomputing them."""
    limit, size = 256 << 20, 0

    def get_or(self, key, fn):
        if key in self:
            self.move_to_end(key)
            return list(self[key]) if isinstance(self[key], list) else self[key]
        val = self[key] = fn()
        self.size += _nbytes(val)
        while self.size > self.limit and len(self) > 1:
            self.size -= _nbytes(self.popitem(last=False)[1])
        return val


_cache = _Cache()


def _digest(*parts):
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        h.update(p.tobytes() if isinstance(p, np.ndarray) else str(p).encode())
    return h.hexdigest()


# ----------------------------------------------------------------------------- audio

def _read_audio(path):
    try:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError) as e:
        raise FileNotFoundError(f"cannot read audio {path!s}: {e}") from None
    return _np(x.mean(1), f"audio {path!s}"), int(sr)


def _audio(audio, cache):
    """Any AudioFile-like input -> (mono float32, sr | None, target | None).
    sr is the rate the data carries; target is a rate the caller attached to a file (tuple / dict)."""
    if isinstance(audio, (str, Path)):
        x, sr = cache.get(str(audio)) or _read_audio(audio)
        return x, sr, None
    if isinstance(audio, dict):
        if "waveform" in audio and "audio" in audio:
            raise ValueError("audio dict has both 'waveform' and 'audio'")
        rate = audio.get("sample_rate")
        rate = None if rate is None else _check_rate(rate)
        if "waveform" in audio:
            x, _, _ = _audio(audio["waveform"], cache)
            return x, rate, None
        x, sr, _ = _audio(audio["audio"], cache)
        return x, sr, rate
    if isinstance(audio, tuple):  # (array | path, sr): the rate of raw data, or the target for a file
        x, sr, _ = _audio(audio[0], cache)
        rate = _check_rate(audio[1])
        return (x, sr, rate) if sr is not None else (x, rate, None)
    x = _np(audio, "audio")
    if x.ndim == 2:  # (C,T) or (T,C): average the short axis
        x = x.mean(0) if x.shape[0] <= x.shape[1] else x.mean(1)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"audio must be a non-empty 1-D or 2-D array, got shape {x.shape}")
    return x, None, None


def _check_rate(sr):
    if isinstance(sr, bool) or not isinstance(sr, (int, float, np.number)) or sr <= 0 or sr != int(sr):
        raise ValueError(f"sample_rate must be a positive integer, got {sr!r}")
    return int(sr)


def _to_rate(x, sr, target):
    """Resample to `target` Hz: polyphase when scipy is present, linear otherwise."""
    if sr == target:
        return x
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, target)
        return resample_poly(x, target // g, sr // g).astype(np.float32)
    except ImportError:
        return _resample(x, round(len(x) * target / sr), "linear")


def _load_audios(audios, sample_rate):
    """[(name, input)] -> [(name, x, sr)] at one rate and one length. Files are read once, in parallel."""
    files = sorted({str(a if not isinstance(a, dict) else a.get("audio", "")) for _, a in audios
                    if isinstance(a, (str, Path)) or (isinstance(a, dict) and isinstance(a.get("audio"), (str, Path)))}
                   | {str(a[0]) for _, a in audios if isinstance(a, tuple) and isinstance(a[0], (str, Path))})
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(files)))) as ex:
        cache = dict(zip(files, ex.map(_read_audio, files))) if files else {}
    loaded = [(name, *_audio(a, cache)) for name, a in audios]
    for name, x, s, t in loaded:
        if s is None:
            raise ValueError(f"audio {name!r} is a bare array: pass (array, sr) or {{'waveform': array, 'sample_rate': sr}}")
    known = {t if t is not None else s for _, _, s, t in loaded}
    sr = _check_rate(sample_rate) if sample_rate is not None else max(known)  # no target: the highest rate, nothing is lost
    out = []
    for name, x, s, t in loaded:
        if t is not None:
            x, s = _to_rate(x, s, t), t
        out.append((name, _to_rate(x, s, sr), sr))
    n = min(len(x) for _, x, _ in out)
    for name, x, _ in out:
        if len(x) - n > max(2, sr // 1000):  # resampling rounds by a sample; more is a real mismatch
            raise ValueError(f"all audios must have the same length at {sr} Hz: {name} has {len(x)} samples, the shortest {n}")
    return [(name, x[:n], sr) for name, x, _ in out]


def _peaks(x, n=N_POINTS):
    """abs-max per bin (every sample covered), so wavesurfer draws the waveform without decoding."""
    k = max(1, -(-len(x) // n))
    return np.abs(np.pad(x, (0, (-len(x)) % k))).reshape(-1, k).max(1).round(3).tolist()


# ----------------------------------------------------------------------------- annotations

def _from_records(recs, uri=""):
    ann = Annotation(uri=uri)
    for d in recs:
        s, e = d.get("start", d.get("start_time")), d.get("end", d.get("end_time"))
        if s is None or e is None:
            raise ValueError(f"record needs start/end (or start_time/end_time): {d}")
        ann[Segment(float(s), float(e))] = str(d.get("speaker", d.get("label", "")))
    return ann


def _from_lines(lines, uri=""):
    """'start end label' lines (.lab, Audacity labels, NeMo Sortformer diarize() output)."""
    ann = Annotation(uri=uri)
    for line in lines:
        f = str(line).split(maxsplit=2)
        if len(f) >= 2:
            ann[Segment(float(f[0]), float(f[1]))] = f[2].strip() if len(f) > 2 else ""
    return ann


def _from_rttm(lines):
    """RTTM / UEM lines -> {uri: Annotation}."""
    out = {}
    for f in (line.split() for line in lines):
        if len(f) >= 8 and f[0] == "SPEAKER":
            out.setdefault(f[1], Annotation(uri=f[1]))[Segment(float(f[3]), float(f[3]) + float(f[4]))] = f[7]
        elif len(f) == 4 and f[0] != "SPEAKER":  # UEM: uri channel start end
            out.setdefault(f[0], Annotation(uri=f[0]))[Segment(float(f[2]), float(f[3]))] = ""
    return out


def _read_annotations(path):
    """Annotation file -> [(name, Annotation)]: .rttm / .uem (one group per uri), .lab / .txt,
    .json / .seglst (records, or {"diarization": [...]}), .jsonl NeMo manifest (follows rttm_filepath)."""
    path = Path(path)
    ext, stem, text = path.suffix.lower(), path.stem, path.read_text()
    if ext in (".rttm", ".uem"):
        out = _from_rttm(text.splitlines())
        return [(stem if len(out) == 1 else f"{stem}:{uri}", a) for uri, a in out.items()]
    if ext == ".jsonl":
        res = []
        for line in filter(str.strip, text.splitlines()):
            p = Path(json.loads(line)["rttm_filepath"])
            res += _read_annotations(p if p.is_absolute() else path.parent / p)
        return res
    if ext in (".json", ".seglst"):
        data = json.loads(text)
        return [(stem, _from_records(data.get("diarization", data) if isinstance(data, dict) else data, stem))]
    return [(stem, _from_lines(text.splitlines(), stem))]


def _matrix(o):
    """Frame-level model output -> (T, K) probabilities, or None. .logits (HF WavLM / Wav2Vec2 frame
    classification) -> sigmoid; (1, T, K) squeezed; (K, T) with K < T transposed; probs pass through."""
    logits = hasattr(o, "logits")
    a = _np(o.logits if logits else o, "function")
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        return None
    if a.shape[0] < a.shape[1]:
        a = a.T
    return 1 / (1 + np.exp(-a)) if logits else a


def _annotation(obj, duration, threshold):
    """Any supported annotation object -> Annotation, segments clipped to [0, duration]."""
    if hasattr(obj, "speaker_diarization"):          # pyannote.audio 4 DiarizeOutput
        ann = obj.speaker_diarization
    elif isinstance(obj, dict) and "diarization" in obj:
        ann = _from_records(obj["diarization"])
    elif isinstance(obj, dict) and "rttm_filepath" in obj:
        ann = _read_annotations(obj["rttm_filepath"])[0][1]
    elif isinstance(obj, str):                       # RTTM text
        if not obj.lstrip().startswith("SPEAKER "):
            raise TypeError(f"string annotation must be RTTM text or a path with a known extension {sorted(ANN_EXT)}: {obj[:40]!r}")
        ann = next(iter(_from_rttm(obj.splitlines()).values()), Annotation())
    elif _is_lines(obj):
        ann = _from_lines(obj)
    elif _is_records(obj):
        ann = _from_records(obj)
    elif _is_tuples(obj):
        ann = _from_lines(" ".join(map(str, e)) for e in obj)
    elif _is_speaker_dict(obj):
        ann = Annotation()
        for lab, segs in obj.items():
            for e in ([segs] if isinstance(segs, Segment) else segs):
                seg = e if isinstance(e, Segment) else Segment(float(e[0]), float(e[1]))
                if seg.end <= seg.start:
                    raise ValueError(f"speaker {lab!r}: segment end must be after start, got {tuple(seg)}")
                ann[seg] = str(lab)
    elif isinstance(obj, Segment):
        ann = _from_lines([f"{obj.start} {obj.end}"])
    elif isinstance(obj, Timeline):
        ann = Annotation(uri=obj.uri)
        for seg in obj:
            ann[seg] = ""
    elif isinstance(obj, Annotation):
        ann = obj
    elif hasattr(obj, "logits") or (_is_array(obj) and np.ndim(obj) in (2, 3)):
        m = _matrix(obj)                             # frames spread over the audio, binarized
        if m is None:
            raise TypeError("frame matrix must be (T, K) or (1, T, K)")
        if m.min() < 0 or m.max() > 1:
            raise ValueError("frame matrix annotation must hold probabilities in [0, 1]; pass an object with .logits for raw logits")
        ann, hop = Annotation(), duration / len(m)
        for k in range(m.shape[1]):
            edges = np.flatnonzero(np.diff(np.r_[0, (m[:, k] > threshold).astype(np.int8), 0]))
            for s, e in edges.reshape(-1, 2):
                ann[Segment(s * hop, e * hop)] = f"spk{k}"
    else:
        raise TypeError(f"unsupported annotation: {type(obj).__name__}")
    clipped = Annotation(uri=ann.uri)
    for seg, track, lab in ann.itertracks(yield_label=True):
        s, e = max(0.0, seg.start), min(duration, seg.end)
        if e > s:
            clipped[Segment(s, e), track] = lab
    return clipped


def _color_specs(colors, form, n):
    """Per-annotation color specs mirroring the annotation form: dict <-> dict by name (subset ok),
    list <-> list per annotation, one <-> one. Returns a dict by name or a list of n specs."""
    if colors is None:
        return [None] * n
    css = lambda v: isinstance(v, str)
    if isinstance(colors, dict) and all(map(css, colors.values())):   # {speaker: css}: a speaker map, for every annotation
        return [colors] * n
    if isinstance(colors, (list, tuple)) and len(colors) > 0 and all(map(css, colors)):  # [css, ...] in labels() order
        return [list(colors)] * n
    if isinstance(form, dict):
        if not isinstance(colors, dict):
            raise ValueError("colors must be a dict keyed by annotation name when annotation is a dict")
        return {str(k): v for k, v in colors.items()}
    if isinstance(form, (list, tuple)) and not _is_ann(form):
        if isinstance(colors, (dict, str)) or len(colors) != n:
            raise ValueError(f"colors must be a list with one entry per annotation ({n})")
        return list(colors)
    if n != 1:
        raise ValueError(f"colors must be given per annotation (dict or list) when there are {n} annotations")
    return [colors]


def _groups(named, colors, form, duration, threshold):
    """One lane group per annotation. Default colors: sorted union of labels -> Set1; per-annotation
    override {speaker: css} by name or a list of css in Annotation.labels() order."""
    anns = []
    for name, obj in named:
        if _is_ann_path(obj):
            read = _read_annotations(obj)
            anns += [(name if name is not None and len(read) == 1 else n, a) for n, a in read]
        else:
            anns.append((name, _annotation(obj, duration, threshold)))
    names = _unique([n for n, _ in anns], "sp")
    specs = _color_specs(colors, form, len(anns))
    labels_all = sorted({str(l) for _, a in anns for l in a.labels()})
    base = {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(labels_all)}
    groups = []
    for i, (name, (_, ann)) in enumerate(zip(names, anns)):
        labels = [str(l) for l in ann.labels()]
        if len(set(labels)) != len(labels):
            raise ValueError(f"annotation {name!r}: labels collide after str(): {labels}")
        cmap = {lab: base[lab] for lab in labels}
        spec = specs.get(name) if isinstance(specs, dict) else specs[i]
        if isinstance(spec, dict):
            cmap.update({str(k): v for k, v in spec.items() if str(k) in cmap})
        elif isinstance(spec, (list, tuple)):
            cmap.update(zip(labels, map(str, spec)))
        elif spec is not None:
            raise ValueError(f"colors for {name!r} must be {{speaker: css}} or a list of css, got {type(spec).__name__}")
        groups.append({
            "name": name,
            "rows": labels,
            "segments": [{"start": s.start, "end": s.end, "label": str(lab), "color": cmap[str(lab)]}
                         for s, _, lab in ann.itertracks(yield_label=True)],
        })
    return groups


# ----------------------------------------------------------------------------- functions & weights

def _resample(y, n, method="linear"):
    """1-D signal of any length -> n points on the same span, frame-centre convention: sample k of L
    sits at (k + 0.5) / L. Downsampling pools (max) so short pulses survive; upsampling interpolates."""
    if method not in METHODS:
        raise ValueError(f"interpolation must be one of {METHODS}, got {method!r}")
    y = _np(y, "signal").reshape(-1)
    L = len(y)
    if L == 0 or n < 1:
        raise ValueError("empty signal")
    if L == n:
        return y
    if L > n:  # pool: strongest excursion per bin (sign kept), so pulses and dips both survive
        return _pool(y, -(-L // n))
    t = (np.arange(n) + 0.5) / n * L  # fractional frame index
    if method == "previous" or L == 1:
        return y[np.minimum(np.floor(t).astype(int), L - 1)]
    if method == "nearest":
        return y[np.clip(np.round(t - 0.5).astype(int), 0, L - 1)]
    src = np.arange(L) + 0.5
    if method == "linear":
        return np.interp(t, src, y).astype(np.float32)
    from scipy.interpolate import CubicSpline
    return np.clip(CubicSpline(src, y)(t), y.min(), y.max()).astype(np.float32)  # no overshoot


def _pad(y, k):
    return np.pad(y, (0, (-len(y)) % k), mode="edge")


def _pool(y, k):
    """Bins of k samples -> the sample with the largest |value| in each bin (sign kept)."""
    b = _pad(y, k).reshape(-1, k)
    return b[np.arange(len(b)), np.abs(b).argmax(1)].astype(np.float32)


def _bin_mean(t, k):
    """Mean time per bin of k, the last partial bin averaged over its real members only."""
    b = np.pad(t.astype(np.float64), (0, (-len(t)) % k), constant_values=np.nan).reshape(-1, k)
    return np.nanmean(b, axis=1).astype(np.float32)


def _weight(x, w, method):
    """Scalar -> x * w; array of len(x) -> elementwise; other length -> interpolated to len(x)."""
    if isinstance(w, (bool, str)) or w is None:
        raise ValueError(f"weight must be a number or an array, got {w!r}")
    if np.ndim(w) == 0:
        if not np.isfinite(float(w)):
            raise ValueError(f"weight must be finite, got {w!r}")
        return x * float(w)
    w = _np(w, "weight")
    if w.ndim != 1:
        raise ValueError(f"weight must be a scalar or 1-D, got shape {w.shape}")
    return x * _resample(w, len(x), method)


def _match(arg, form, names, what):
    """Pair a per-audio argument with the audios: dict <-> dict (same keys), list <-> list (same length), one <-> one."""
    if isinstance(form, dict):
        if not isinstance(arg, dict) or set(map(str, arg)) != set(names):
            raise ValueError(f"{what} must be a dict with the same keys as audio: {names}")
        arg = {str(k): v for k, v in arg.items()}
        return [arg[n] for n in names]
    if isinstance(form, (list, tuple)) and not _is_audio(form):
        if isinstance(arg, (dict, str)) or np.ndim(arg) == 0 or len(arg) != len(names):
            raise ValueError(f"{what} must be a list with one entry per audio ({len(names)})")
        return list(arg)
    if len(names) != 1:
        raise ValueError(f"{what} must be given per audio (dict or list) when there are {len(names)} audios")
    return [arg]


def _bind_functions(function, audio_names):
    """-> [(name, f, row)]: dict keys equal to an audio name bind to that audio's row (value: one
    function, a list, or a {name: f} dict); everything else is drawn on every row (row = -1)."""
    out = []
    if isinstance(function, dict):
        for key, val in function.items():
            key = str(key)
            row = audio_names.index(key) if key in audio_names else -1
            subs = _named(val, _is_fn)
            out += [(f"{key}.{n}" if n else (f"{key}[{j}]" if len(subs) > 1 else key), f, row) for j, (n, f) in enumerate(subs)]
    else:
        out += [(n, f, -1) for n, f in _named(function, _is_fn)]
    for n, f, _ in out:
        if not _is_fn(f):
            raise TypeError(f"function {n or ''!r}: expected an array, tensor, SlidingWindowFeature or .logits, got {type(f).__name__}")
    return out


def _functions(bound, duration, method, n=N_POINTS):
    """-> [{"name", "color", "row", "x": 0..1 of duration, "y": 0..1 min-max scaled}, ...]"""
    out = []
    for name, f, row in bound:
        if isinstance(f, SlidingWindowFeature):  # real time axis from the window middles
            data = _np(f.data, f"function {name!r}")
            if data.ndim not in (1, 2) or len(data) == 0:
                raise ValueError(f"function {name!r}: SlidingWindowFeature data must be (T,) or (T, K) and non-empty, got {data.shape}")
            t = np.array([f.sliding_window[i].middle for i in range(len(data))], dtype=np.float32) / duration
            keep = t <= 1.0
            if not keep.any():
                raise ValueError(f"function {name!r}: every frame lies past the audio ({duration:.2f} s)")
            cols = [c[keep] for c in ([data] if data.ndim == 1 else list(data.T))]
            t = t[keep]
            if len(t) > n:  # pool like the raw path: the strongest excursion per bin, at the bin's mean time
                k = -(-len(t) // n)
                cols, t = [_pool(c, k) for c in cols], _bin_mean(t, k)
        else:
            m = _matrix(f) if (hasattr(f, "logits") or np.ndim(f) >= 2) else None
            raw = list(m.T) if m is not None else [_np(f, f"function {name!r}").reshape(-1)]
            L = len(raw[0])
            cols = [_resample(c, n, method) for c in raw]
            t = (np.arange(L) + 0.5) / L            # frame centres
            t = _bin_mean(t, -(-L // n)) if L > n else (np.arange(n) + 0.5) / n
        names = [name] if len(cols) == 1 else [f"{name or ''}[{k}]" for k in range(len(cols))]
        for nm, y in zip(names, cols):
            lo, hi = float(y.min()), float(y.max())
            y = (y - lo) / (hi - lo) if hi > lo else np.full_like(y, 0.5)
            out.append({"name": nm or "", "color": LINE_PALETTE[len(out) % len(LINE_PALETTE)], "row": row,
                        "x": np.round(t, 6).tolist(), "y": np.round(y, 3).tolist()})
    return out


# ----------------------------------------------------------------------------- spectrogram

def _mel_rows(db, sr, height, fmax, n_bins):
    """(bins, cols) linear-frequency dB (bins up to fmax, out of n_bins for the full band) -> (height, cols)
    on a mel axis up to fmax (low frequencies get most of the rows, as librosa's y_axis="mel"); each row
    takes the max over its bins, so tones survive."""
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    f_edges = 700 * (10 ** (np.linspace(mel(0), mel(fmax), height + 1) / 2595) - 1)
    b = np.clip(np.round(f_edges / (sr / 2) * (n_bins - 1)).astype(int), 0, db.shape[0] - 1)
    return np.stack([db[lo:max(hi, lo + 1)].max(0) for lo, hi in zip(b[:-1], b[1:])])


def _cmap(cmap):
    """matplotlib colormap name (case-insensitive) or Colormap object -> (Colormap, key for the cache)."""
    from matplotlib import colormaps
    from matplotlib.colors import Colormap
    if isinstance(cmap, Colormap):
        return cmap, f"cmap:{cmap.name}:{id(cmap)}"
    if isinstance(cmap, str):
        name = next((k for k in colormaps if k.lower() == cmap.lower()), None)
        if name is not None:
            return colormaps[name], name
    raise ValueError(f"unknown colormap {cmap!r}; any matplotlib name or Colormap object works, e.g. magma, viridis, gray_r")


def _spectrogram(x, sr, cmap, height, fscale="mel", fmax=None, n_fft=None, max_cols=4000, dyn_range=80.0, floor_dbfs=-60.0):
    """Log-magnitude STFT as a JPEG data URL, `height` rows (frequency axis mel by default, or linear;
    high frequencies on top), <= max_cols columns. Every sample is observed: frames step n_fft/4 and
    are max-pooled into columns, computed in chunks so memory stays bounded. Column i is centred at
    (i + 0.5) / cols of the duration. The dB reference is max(peak, floor): silence stays dark."""
    try:
        from matplotlib import image
        from PIL import Image
    except ImportError:
        return ""
    cmap, _ = _cmap(cmap)
    if fscale not in ("mel", "linear"):
        raise ValueError(f"fscale must be 'mel' or 'linear', got {fscale!r}")
    if fmax is not None and (isinstance(fmax, (bool, str)) or not isinstance(fmax, (int, float, np.number))):
        raise ValueError(f"fmax must be a number in Hz, got {fmax!r}")
    fmax = sr / 2 if fmax is None else float(fmax)
    if not 0 < fmax <= sr / 2:
        raise ValueError(f"fmax must be in (0, {sr / 2:.0f}] Hz, got {fmax}")
    if n_fft is None:  # ~32 ms window at any rate: 512 @ 16 kHz, 2048 @ 48 kHz (same Hz per bin on the mel rows)
        n_fft = int(min(4096, max(256, 2 ** round(math.log2(0.032 * sr)))))
    hop = n_fft // 4
    win = np.hanning(n_fft).astype(np.float32)
    gain = win.sum() / 2  # |rfft| of a full-scale sine == gain
    n = len(x)
    n_frames = -(-n // hop)
    cols = min(max_cols, n_frames)
    # frame f is centred at (f + 0.5) * hop samples; column i covers [i, i + 1) * n / cols samples, so the
    # image spans exactly the audio (no n_frames * hop overhang on short clips)
    col_of_frame = np.minimum(((np.arange(n_frames) * hop + hop // 2) * cols) // n, cols - 1)
    out = np.zeros((n_fft // 2 + 1, cols), np.float32)
    xp = np.pad(x, (n_fft // 2 - hop // 2, n_fft))  # frame f is centred at (f + 0.5) * hop, like its column
    frames_per_chunk = (1 << 20) // hop
    for f0 in range(0, n_frames, frames_per_chunk):
        f1 = min(n_frames, f0 + frames_per_chunk)
        frames = np.lib.stride_tricks.sliding_window_view(xp[f0 * hop: (f1 - 1) * hop + n_fft], n_fft)[::hop] * win
        mag = np.abs(np.fft.rfft(frames, axis=1)).T / gain
        c = col_of_frame[f0:f1]
        pooled = np.maximum.reduceat(mag, np.flatnonzero(np.r_[1, np.diff(c)]), axis=1)
        out[:, c[0]:c[-1] + 1] = np.maximum(out[:, c[0]:c[-1] + 1], pooled)
    ref = max(float(out.max()), 10 ** (floor_dbfs / 20))
    db = np.clip(20 * np.log10(out / ref + 1e-12), -dyn_range, 0) / dyn_range + 1
    n_bins = db.shape[0]
    db = db[: max(2, int(round(fmax / (sr / 2) * (n_bins - 1))) + 1)]  # bins up to fmax
    if fscale == "mel":
        db = _mel_rows(db, sr, height, fmax, n_bins)
    elif height < db.shape[0]:  # shrink by max-pooling so a 1-bin tone keeps its brightness
        edges = (np.arange(height) * db.shape[0]) // height
        db = np.maximum.reduceat(db, edges, axis=0)
    else:
        db = np.asarray(Image.fromarray(db, mode="F").resize((cols, height), Image.BILINEAR))
    buf = io.BytesIO()
    image.imsave(buf, cmap(db[::-1])[..., :3], format="jpeg", pil_kwargs={"quality": 85, "subsampling": 0})  # 4:4:4: thin lines keep their colour
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def specs_possible():
    try:
        import matplotlib, PIL  # noqa: F401
        return True
    except ImportError:
        return False


def _mix_spec(on, sr, cmap, height, fscale, fmax):
    """Spectrogram of the mean of the selected audios (what is played), cached by content."""
    mix = sum(on) / len(on)
    _, ck = _cmap(cmap)
    return _cache.get_or(_digest(_digest(mix, sr), ck, height, fscale, fmax), lambda: _spectrogram(mix, sr, cmap, height, fscale, fmax))


# ----------------------------------------------------------------------------- widget

class WavLens(anywidget.AnyWidget):
    _esm = Path(__file__).parent / "wavlens.js"
    audio_url = traitlets.Unicode("").tag(sync=True)
    duration = traitlets.Float(0.0).tag(sync=True)
    groups = traitlets.List([]).tag(sync=True, echo_update=False)  # [{"name", "rows", "segments"}, ...]; the browser reorders by drag
    expanded = traitlets.List(traitlets.Unicode()).tag(sync=True, echo_update=False)  # annotation names whose legend column is open
    order = traitlets.List(traitlets.Int()).tag(sync=True, echo_update=False)        # display order of the audio rows (original indices); drag to change
    functions = traitlets.List([]).tag(sync=True)       # [{"name", "color", "row", "x", "y"}, ...]; row = audio index or -1
    peaks = traitlets.List([]).tag(sync=True)           # per audio (+ the mix row when >= 2 audios): abs-max bins
    spectrograms = traitlets.List([]).tag(sync=True)    # per audio (+ mix): JPEG data URL, stretched to the row
    mix_mask = traitlets.List([]).tag(sync=True)        # the enabled set the last spectrogram (the mix) was computed for
    mix_specs = traitlets.Dict({}).tag(sync=True)       # {"1101": url}: mix spectrograms per enabled set (all subsets for <= 3 audios, then every set seen)
    audio_names = traitlets.List([]).tag(sync=True)
    channels = traitlets.Int(1).tag(sync=True)
    weighted = traitlets.Bool(False).tag(sync=True)     # weights baked in -> gains 1/0; else equal 1/n live
    height = traitlets.Int(96).tag(sync=True)           # px per audio row
    row_height = traitlets.Int(16).tag(sync=True)       # px per annotation label row
    view = traitlets.Enum(["wave", "fft"], "wave").tag(sync=True)
    step = traitlets.Float(0.01).tag(sync=True)         # seconds per ◀ / ▶ press
    zoom = traitlets.Float(0.0).tag(sync=True)          # px per second; 0 = fit
    autoplay = traitlets.Bool(False).tag(sync=True)
    # Browser-driven traits: echo_update=False, otherwise the kernel echoes stale values back over a
    # slow link and the widget seeks backwards / toggles play again.
    current_time = traitlets.Float(0.0).tag(sync=True, echo_update=False)
    playing = traitlets.Bool(False).tag(sync=True, echo_update=False)
    enabled = traitlets.List().tag(sync=True, echo_update=False)  # bools, cast in the validator
    solo = traitlets.List(traitlets.Unicode()).tag(sync=True, echo_update=False)  # "g:<i>" | "l:<i>:<spk>", "!" prefix = complement; active keys intersect
    dimmed = traitlets.List(traitlets.Unicode()).tag(sync=True, echo_update=False)

    @traitlets.validate("height", "row_height")
    def _positive_int(self, p):
        if p["value"] <= 0:
            raise traitlets.TraitError(f"{p['trait'].name} must be > 0")
        return p["value"]

    @traitlets.validate("step", "zoom", "current_time")
    def _finite(self, p):
        v = float(p["value"])
        if not np.isfinite(v) or v < 0 or (p["trait"].name == "step" and v == 0):
            raise traitlets.TraitError(f"{p['trait'].name} must be finite and {'> 0' if p['trait'].name == 'step' else '>= 0'}")
        return v

    @traitlets.validate("expanded")
    def _expanded_names(self, p):
        names = [g["name"] for g in self.groups]
        out = []
        for n in p["value"]:
            if n not in names:
                raise traitlets.TraitError(f"unknown annotation {n!r}; have {names}")
            if n not in out:
                out.append(n)
        return out

    @traitlets.validate("groups")
    def _groups_shape(self, p):
        v = p["value"]
        if not all(isinstance(g, dict) and {"name", "rows", "segments"} <= set(g) for g in v):
            raise traitlets.TraitError("groups must be a list of {name, rows, segments} dicts (reorder w.groups, do not rebuild it)")
        return v

    @traitlets.observe("groups")
    def _groups_reordered(self, change):
        """Reassigning w.groups (e.g. a new order) keeps solo/dimmed/expanded pointing at the same names."""
        old = [g["name"] for g in (change["old"] or [])]
        if not old or old == [g["name"] for g in self.groups]:
            return
        new = {g["name"]: i for i, g in enumerate(self.groups)}
        rows = {g["name"]: set(g["rows"]) for g in self.groups}

        def remap(key):
            inv, k = key.startswith("!"), key.lstrip("!")
            kind, i, *rest = k.split(":")
            name = old[int(i)] if int(i) < len(old) else None
            if name not in new or (rest and rest[0] not in rows[name]):
                return None
            return ("!" if inv else "") + ":".join([kind, str(new[name]), *rest])
        self.solo = [k for k in map(remap, self.solo) if k]
        self.dimmed = [k for k in map(remap, self.dimmed) if k]
        self.expanded = [n for n in self.expanded if n in new]

    def close(self):
        self._mix = None  # release the kept audio copies
        super().close()

    @traitlets.validate("order")
    def _order_perm(self, p):
        v = [int(i) for i in p["value"]]
        if sorted(v) != list(range(self.channels)):
            raise traitlets.TraitError(f"order must be a permutation of range({self.channels}), got {v}")
        return v

    @traitlets.validate("enabled")
    def _enabled_len(self, p):
        v = p["value"]
        if isinstance(v, (str, dict)) or not all(isinstance(e, (bool, np.bool_, int, np.integer)) for e in v):
            raise traitlets.TraitError("enabled must be a list of bools")
        if self.channels and len(v) != self.channels:
            raise traitlets.TraitError(f"enabled needs {self.channels} entries, got {len(v)}")
        return [bool(e) for e in v]  # numpy bools are not JSON-serializable

    _mix = None  # (xs, sr, colormap, height, fscale, fmax) when there are >= 2 audios

    @traitlets.observe("enabled")
    def _mix_changed(self, _):
        """Keep the mix row's spectrogram in step with the enabled set (the browser redraws its waveform itself)."""
        if not self._mix:
            return
        xs, sr, cmap, height, fscale, fmax = self._mix
        on = [x for x, e in zip(xs, self.enabled) if e]
        if len(on) < 2:
            return  # no mix row shown
        key = "".join("1" if e else "0" for e in self.enabled)
        spec = self.mix_specs.get(key) or _mix_spec(on, sr, cmap, height, fscale, fmax)
        if key not in self.mix_specs:
            self.mix_specs = {**self.mix_specs, key: spec}  # exports made later carry every subset seen
        if not self.spectrograms or self.spectrograms[-1] != spec or self.mix_mask != list(self.enabled):
            self.spectrograms = self.spectrograms[:len(xs)] + [spec]
            self.mix_mask = list(self.enabled)

    def seek(self, t):
        self.current_time = float(t)

    def play(self):
        self.playing = True

    def pause(self):
        self.playing = False

    def _key(self, item):
        """'annotation' | 'annotation/speaker' | ('annotation', 'speaker') -> JS key."""
        if isinstance(item, (tuple, list)) and len(item) not in (1, 2):
            raise ValueError(f"key must be 'annotation', 'annotation/speaker' or (annotation, speaker), got {item!r}")
        name, speaker = (tuple(map(str, item)) + ("",))[:2] if isinstance(item, (tuple, list)) else (str(item).split("/", 1) + [""])[:2]
        names = [g["name"] for g in self.groups]
        if name not in names:
            raise ValueError(f"unknown annotation {name!r}; have {names}")
        g = self.groups[names.index(name)]
        if speaker and speaker not in g["rows"]:
            raise ValueError(f"unknown speaker {speaker!r} in {name!r}; have {g['rows']}")
        return f"l:{names.index(name)}:{speaker}" if speaker else f"g:{names.index(name)}"

    def solo_(self, *items):
        """Mask playback to annotations ('ref' = union of its speakers) and/or speakers ('ref/speaker90',
        ('ref', 'speaker90')); a leading '!' keeps the complement ('!ref/speaker90'). Several items
        intersect. Gaps are silenced, whited out and skipped. No arguments clears."""
        keys = []
        for i in items:
            inv = isinstance(i, str) and i.startswith("!")
            keys.append(("!" if inv else "") + self._key(i[1:] if inv else i))
        self.solo = keys

    def dim(self, *items):
        """Dim these annotations / speakers; no arguments clears."""
        self.dimmed = [self._key(i) for i in items]

    # ---- standalone export: the same bundle driven by an in-page model, no kernel (Comet, W&B, email)
    def html(self, title="wavlens"):
        """Self-contained HTML string: audio, spectrograms, lanes, functions and every control work; only
        the Python sync is absent. Use for experiment trackers, reports, e-mail."""
        js = (Path(__file__).parent / "wavlens.iife.js").read_text()
        state = json.dumps({n: getattr(self, n) for n in WavLens.class_own_traits(sync=True)})  # declared class: anywidget subclasses per instance
        return f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{margin:0;padding:8px;background:#fff;font-family:sans-serif}}</style></head><body><div id="wavlens"></div>
<script>{js}</script>
<script>
(function(){{const state={state},subs={{}};
const model={{get:(k)=>state[k],set:(k,v)=>{{state[k]=v;(subs["change:"+k]||[]).forEach((f)=>f());}},
  save_changes(){{}},on:(e,f)=>{{(subs[e]=subs[e]||[]).push(f);}},off:(e,f)=>{{subs[e]=(subs[e]||[]).filter((g)=>g!==f);}}}};
wavlens_mod.default.render({{model,el:document.getElementById("wavlens")}});}})();
</script></body></html>"""

    def save_html(self, path, title="wavlens"):
        Path(path).write_text(self.html(title))
        return Path(path)

    def log_comet(self, experiment, name="wavlens", step=None):
        """Comet ML: `experiment.log_html` (HTML tab); call once per epoch with a different name/step."""
        experiment.log_html(self.html(name))
        experiment.log_asset_data(self.html(name), name=f"{name}{'' if step is None else f'_{step}'}.html", step=step)

    def log_wandb(self, key="wavlens", step=None):
        """Weights & Biases: `wandb.log({key: wandb.Html(...)})`, renders in the run page."""
        import wandb
        wandb.log({key: wandb.Html(self.html(key), inject=False)}, step=step)


# ----------------------------------------------------------------------------- show

def show(*items, audio=None, annotation=None, function=None, sample_rate=None, height=96, row_height=16,
         weights=None, interpolation="linear", colors=None, colormap="magma", view="wave", expanded=True,
         step=0.01, enabled=None, solo=None, dim=(), zoom=0.0, time=0.0, autoplay=False, threshold=0.5,
         fscale="mel", fmax=None, order=None):
    """wavlens.show(audio, annotation, ...) positional in any order, or by keyword.

    audio:      path | pyannote AudioFile dict {"audio": path} or {"waveform": array, "sample_rate": sr}
                | (array, sr) (what torchaudio.load returns; (path, sr) resamples the file to sr); or a
                list / {name: ...} dict of those. Every audio carries its rate: a bare array raises. One
                waveform row each (unnamed: Audio / Audio_<i>), played together. Integers are scaled to
                ±1, any torch device/dtype is accepted, NaN/inf raise.
    sample_rate: optional target rate; every audio is resampled to it. Default None: audios with
                different rates are resampled to the highest one (nothing is lost), so all share one
                time line. Lengths must then match (resampling rounding of a sample or so is trimmed).
    annotation: as object: pyannote Annotation / Timeline / Segment / DiarizeOutput, its serialize()
                dict, NeMo Sortformer diarize() output (list of "start end speaker" strings), a NeMo
                manifest dict, a list of {start, end, speaker} records (SegLST start_time/end_time,
                WhisperX), (start, end, label) tuples, RTTM text, an HF ModelOutput with .logits
                (WavLM), or (keyword only, since positional 2-D arrays are audio) a frame matrix
                (T, K) such as Nemotron-3 Sortformer probs, binarized at `threshold`. As path: .rttm,
                .uem, .lab/.txt, .json/.seglst, NeMo manifest .jsonl. Or a list / {name: ...} dict of
                those. One lane group each (unnamed: sp / sp_<i>; files: the file stem, an RTTM with
                several uris gives one group per uri). Segments are clipped to the audio.
    function:   1-D signal, a frame matrix (T, K) / (1, T, K) or an HF ModelOutput (sigmoid of the
                logits, one curve per column), or a pyannote SlidingWindowFeature; or a list / {name:
                ...} dict. Min-max scaled to 0..1. Dict keys equal to an audio name bind those
                functions to that audio's row only; other keys are general and go on every row.
                Time convention everywhere: frame k of L sits at (k + 0.5) / L of the duration.
    weights:    per audio, mirroring the audio form (dict <-> dict same keys, list <-> list, one <->
                one). Scalar, array of the audio's length, or any other length (interpolated).
                Applied sample-exactly in Python (you see and hear w * audio). Playback = sum over the m
                selected audios of w_i * audio_i / m: selecting or muting re-normalizes by 1/m live. With
                >= 2 audios a framed, non-selectable "mix" row shows that sum above the timeline.
    interpolation: linear | nearest | previous | cubic (clipped, no overshoot) for weights and functions
                of other lengths. Downsampling keeps the strongest excursion per bin (sign kept) so pulses
                and dips survive. Under the frame-centre convention nearest and previous coincide except
                at exact ties. A dim/solo tuple always means (annotation, speaker); use a list for several.
    colors:     per annotation, mirroring the annotation form; each entry {speaker: css} by name or a
                list of css in Annotation.labels() order. Default: sorted union of labels -> Set1.
    colormap:   any matplotlib name. fscale: "mel" (default, low band gets most rows) | "linear".
                fmax: top of the frequency axis in Hz (default Nyquist), e.g. 5000 for speech.
    view:       "wave" | "fft". expanded: annotation names whose speaker legend column starts open
                (True = all); the ▸ button beside each name toggles it in the browser. Drag a name up or
                down to reorder the lane groups (the order is reflected in w.groups); drag an audio's
                checkbox pill to reorder the audio rows (`order`, a permutation of the original indices).
                The mix row is shown while at least two audios are selected.
    height / row_height: px per audio row / per annotation label row. step: seconds per ◀ / ▶.
    enabled / solo ('ann', 'ann/speaker', '!ann/speaker' for the complement, or a list; several
                intersect) / dim / zoom (px per second, 0 = fit) / time / autoplay:
                initial UI state. threshold: binarization level for frame-matrix annotations."""
    if interpolation not in METHODS:
        raise ValueError(f"interpolation must be one of {METHODS}, got {interpolation!r}")
    for nm, v in (("height", height), ("row_height", row_height)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0:
            raise ValueError(f"{nm} must be a positive int, got {v!r}")
    if not isinstance(threshold, (int, float, np.number)) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be a number in [0, 1], got {threshold!r}")
    if enabled is not None and (isinstance(enabled, (str, dict)) or not all(isinstance(e, (bool, np.bool_, int)) for e in enabled)):
        raise ValueError("enabled must be a list of bools, one per audio")
    if isinstance(autoplay, str):
        raise ValueError(f"autoplay must be a bool, got {autoplay!r}")
    for a in items:
        if isinstance(a, tuple) and len(a) == 2 and not _is_audio(a) and not _is_ann(a):
            raise ValueError(f"(audio, sample_rate) tuple needs a numeric rate, got {a[1]!r}")
    flat = []  # a positional list / dict whose items are all audios or all annotations is unpacked
    for a in items:
        if not _is_audio(a) and not _is_ann(a) and isinstance(a, (list, tuple, dict)) and len(a):
            vals = list(a.values()) if isinstance(a, dict) else list(a)
            if all(map(_is_audio, vals)):
                audio = {**(audio or {}), **a} if isinstance(a, dict) and (audio is None or isinstance(audio, dict)) else audio
                if not isinstance(a, dict): flat += vals
                continue
            if all(map(_is_ann, vals)):
                annotation = {**(annotation or {}), **a} if isinstance(a, dict) and (annotation is None or isinstance(annotation, dict)) else annotation
                if not isinstance(a, dict): flat += vals
                continue
        flat.append(a)
    items = flat
    pos_audio = [a for a in items if _is_audio(a)]
    pos_ann = [a for a in items if _is_ann(a)]
    rest = [type(a).__name__ for a in items if not _is_audio(a) and not _is_ann(a)]
    if rest:
        raise TypeError(f"unrecognized positional argument(s): {rest}")
    audios = [(None, a) for a in pos_audio] + _named(audio, _is_audio)
    if not audios:
        raise ValueError("show() needs at least one audio")
    names = _unique([n for n, _ in audios], "Audio")
    loaded = _load_audios([(n, a) for n, (_, a) in zip(names, audios)], sample_rate)
    sr, duration = loaded[0][2], len(loaded[0][1]) / loaded[0][2]
    if not 0 <= float(time) <= duration:
        raise ValueError(f"time must be within the audio (0..{duration:.2f} s), got {time}")
    form = lambda kw, pos: kw if kw is not None else (pos[0] if len(pos) == 1 else pos)
    if weights is not None:
        ws = _match(weights, form(audio, pos_audio), names, "weights")
        loaded = [(n, _weight(x, w, interpolation), s) for (n, x, s), w in zip(loaded, ws)]
    peak = max(float(np.abs(x).max()) for _, x, _ in loaded) or 1.0  # one global gain keeps relative loudness
    xs = [x / peak for _, x, _ in loaded]

    def wav():
        buf = io.BytesIO()
        sf.write(buf, np.stack(xs, 1), sr, format="WAV", subtype="PCM_16")
        return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()
    rows = xs + ([sum(xs) / len(xs)] if len(xs) >= 2 else [])  # mix row = mean of the enabled audios (all at start)
    keys = [_digest(x, sr) for x in rows]
    audio_url = _cache.get_or(_digest(*keys[:len(xs)], "wav"), wav)
    peaks = [_cache.get_or(k + "peaks", lambda x=x: _peaks(x)) for k, x in zip(keys, rows)]
    _, ck = _cmap(colormap) if specs_possible() else (None, str(colormap))
    specs = [_cache.get_or(_digest(k, ck, height, fscale, fmax), lambda x=x: _spectrogram(x, sr, colormap, height, fscale, fmax)) for k, x in zip(keys[:len(xs)], xs)]
    en = [bool(e) for e in enabled] if enabled is not None else [True] * len(loaded)
    mix_specs, mix_mask = {}, []
    if len(xs) >= 2 and specs[0]:
        masks = [[bool(b & (1 << i)) for i in range(len(xs))] for b in range(1 << len(xs))] if len(xs) <= 3 else []
        if len(en) == len(xs) and sum(en) >= 2 and en not in masks:
            masks.append(en)
        if all(en) is False and [True] * len(xs) not in masks:
            masks.append([True] * len(xs))
        for m in masks:
            if sum(m) >= 2:
                mix_specs["".join("1" if b else "0" for b in m)] = _mix_spec([x for x, b in zip(xs, m) if b], sr, colormap, height, fscale, fmax)
        mix_mask = en if len(en) == len(xs) and sum(en) >= 2 else [True] * len(xs)
        specs = specs + [mix_specs["".join("1" if b else "0" for b in mix_mask)]]
    w = WavLens(
        audio_url=audio_url, duration=duration, peaks=peaks, spectrograms=specs, audio_names=names, mix_specs=mix_specs, mix_mask=mix_mask,
        channels=len(loaded), weighted=weights is not None,
        groups=_groups([(None, a) for a in pos_ann] + _named(annotation, _is_ann), colors, form(annotation, pos_ann), duration, threshold),
        functions=_functions(_bind_functions(function, names), duration, interpolation),
        height=height, row_height=row_height, view=view, step=step, zoom=float(zoom),
        current_time=float(time), autoplay=bool(autoplay),
    )
    if len(loaded) >= 2:
        w._mix = (xs, sr, colormap, height, fscale, fmax)
    w.enabled = [bool(e) for e in enabled] if enabled is not None else [True] * len(loaded)
    w.order = list(range(len(loaded))) if order is None else [int(i) for i in order]
    names_g = [g["name"] for g in w.groups]
    if expanded is True:
        w.expanded = names_g
    elif expanded:
        w.expanded = [expanded] if isinstance(expanded, str) else [str(n) for n in expanded]
    if solo is not None and solo != [] and solo != "":
        w.solo_(*(solo if isinstance(solo, list) else [solo]))
    if dim:
        w.dim(*(dim if isinstance(dim, list) else [dim]))
    mb = (len(audio_url) + sum(map(len, specs))) / 1e6
    if mb > STATE_WARN_MB:
        warnings.warn(f"widget state is {mb:.0f} MB (audio {duration:.0f} s at {sr} Hz); it is stored in the notebook output. "
                      "Show a shorter excerpt or a lower sample_rate.", stacklevel=2)
    return w
