# wavlens functionalities

Complete reference of what `wavlens.show(...)` accepts and what the widget does. Written for agents: every form is listed, nothing is implied. Source of truth: `wavlens/__init__.py` (Python) and `js/wavlens.js` (browser, bundled into `wavlens/wavlens.js`).

## 1. Call forms

```python
wavlens.show(*items, audio=None, annotation=None, function=None, sample_rate=None,
           height=96, row_height=16, weights=None, interpolation="linear", colors=None,
           colormap="magma", view="wave", legend="inline", step=0.01,
           enabled=None, solo=None, dim=(), zoom=0.0, time=0.0, autoplay=False, threshold=0.5) -> WavLens
```

- Positional `*items`: any mix of audios and annotations, any order. Dispatch by type: pyannote objects, annotation files, annotation objects (section 3) are annotations; everything else is audio. A positional 2-D array is audio `(C, T)`; frame matrices must go through `annotation=` / `function=`. Anything unrecognized raises `TypeError`.
- Keywords `audio=`, `annotation=`, `function=` each take one item, a list, or a `{name: item}` dict. Positional and keyword items are concatenated.
- Returns a `WavLens` anywidget. Last expression of a cell renders it; `display(w)` elsewhere.

## 2. Audio inputs (`audio=` or positional)

| Form | Rate it carries | Notes |
|---|---|---|
| `"path.wav"` / `Path` | file rate | any libsndfile format; multichannel averaged to mono |
| `{"audio": path}` | file rate | pyannote AudioFile |
| `{"audio": path, "sample_rate": sr}` | file rate → resampled to `sr` | |
| `{"waveform": (C,T) tensor/array, "sample_rate": sr}` | `sr` | pyannote AudioFile; `sample_rate` optional |
| `(array_or_tensor, sr)` | `sr` | what `torchaudio.load()` returns; `(C,T)` or `(T,)` |
| `(path, sr)` | file rate → resampled to `sr` | |
| list / dict of the above | per item | dict keys are the row names |

Rules:
- `_np`: tensors go `.detach().cpu()`, bf16/f16 → float32; integer dtypes divided by their max (int16 → ±1); bool → float; NaN or inf anywhere → `ValueError` with the count; empty, 0-d or 3-D → `ValueError`.
- Every audio carries its rate; a bare `np.ndarray` / `torch.Tensor` raises (`ValueError` naming the audio). `sample_rate=` is an optional target: every audio is resampled to it (scipy `resample_poly`, linear fallback). Without it, audios with different rates are resampled to the highest rate (nothing lost), one time line. Rates must be positive integers.
- All audios must have the same length at the target rate; a difference up to `max(2, sr/1000)` samples (resampling rounding) is trimmed, more raises.
- Files are read once per call even if referenced several times, in parallel (thread pool, up to 8).
- Names: dict keys, else `Audio` (one) / `Audio_<i>`; collisions get `~1`, `~2` suffixes. Non-string keys are `str()`-ed.
- Playback WAV: all audios stacked as channels, 16-bit PCM, divided by one global peak so relative loudness is kept, base64 data URL. Warning above 20 MB of state.

## 3. Annotation inputs (`annotation=` or positional)

As objects:

| Form | Detection |
|---|---|
| `pyannote.core.Annotation` | type |
| `Timeline` | type; one row, label `""` |
| `Segment` | type; one row |
| `DiarizeOutput` (pyannote.audio 4) or anything with `.speaker_diarization` | attribute |
| `{"diarization": [{start, end, speaker}], ...}` (`DiarizeOutput.serialize()`) | key |
| `{"rttm_filepath": ...}` (NeMo manifest entry) | key; the RTTM is read |
| `["0.00 4.98 speaker_0", ...]` (NeMo Sortformer `diarize()`) | list of `"start end label"` strings; a list of such lists = one group per file |
| `[{"start"/"start_time", "end"/"end_time", "speaker"/"label"}, ...]` (SegLST, WhisperX) | list of dicts |
| `[(start, end, label), ...]` or `[(start, end), ...]` | list of tuples with numeric first element |
| RTTM text (`"SPEAKER ...\n..."`) | string with newline starting with `SPEAKER ` |
| HF `ModelOutput` / anything with `.logits` `(B,T,K)` (WavLM frame classification) | attribute; sigmoid, binarized at `threshold` |
| frame matrix `(T,K)` or `(1,T,K)` array/tensor (Nemotron-3 Sortformer probs) | keyword only; `(K,T)` with K < T is transposed; values outside [0,1] get a sigmoid; binarized at `threshold`; labels `spk<k>`; frames spread evenly over the audio |

As paths (by extension): `.rttm` (one group per uri), `.uem` (one row, scoring regions), `.lab` / `.txt` (`start end label` lines, Audacity labels), `.json` / `.seglst` (records or `{"diarization": [...]}`), `.jsonl` (NeMo manifest, follows `rttm_filepath`, relative to the manifest).

Rules:
- Each item becomes one lane group; names: dict key, else file stem (`stem:uri` for multi-uri RTTM), else `sp` / `sp_<i>`.
- One lane row per label in `Annotation.labels()` order (sorted). Labels are `str()`-ed; two labels colliding after `str()` raise.
- Segments are clipped to `[0, duration]`; empty ones dropped.
- Overlapping segments with the same label share the row.

## 4. Functions (`function=`)

Forms: 1-D array/tensor; frame matrix `(T,K)` / `(1,T,K)` or `.logits` (one curve per column, named `name[k]`); `pyannote.core.SlidingWindowFeature` (its own window middles as time axis, columns as curves); list; `{name: item}` dict.

- Dict keys equal to an audio name bind their functions to that audio's row only; the value may be one function, a list (`key[j]`) or a dict (`key.sub`). Other keys are general and are drawn on every audio row and every lane group.
- Each curve is min-max scaled to 0..1 (constant → 0.5), resampled to 2000 points: downsampling max-pools (short pulses survive), upsampling uses `interpolation`.
- NaN/inf raise. Frames of a SlidingWindowFeature past the audio end are dropped.
- Colors cycle `#000000, #1f77b4, #d62728, #2ca02c, #9467bd, #8c564b`; the name is the hover title.

## 5. Weights (`weights=`)

Mirror the `audio` form: dict ↔ dict with the same keys, list ↔ list of the same length, one ↔ one. Each entry: scalar (whole audio), array of the audio's length (elementwise), or any other 1-D length (interpolated to the audio with `interpolation`). Applied sample-exactly in Python before peak normalization: you see and hear `w · audio`. Playback = Σ over the m selected audios of `w_i · audio_i / m`: selecting or muting re-normalizes by 1/m live, with or without explicit weights. Negative weights invert, NaN/None/bool raise, 2-D raise.

## 6. Interpolation

`interpolation="linear" | "nearest" | "previous" | "cubic"`, validated eagerly. Applies to weights and functions whose length differs from the target. Convention: sample k of L sits at `(k + 0.5) / L` of the span. `nearest` rounds half to even, `previous` holds the frame, `cubic` is clipped to the data range (no overshoot). Downsampling always max-pools regardless of method.

## 7. Colors

`colors=` mirrors the `annotation` form: dict keyed by annotation name (subset ok, unknown keys ignored), list per annotation, or one entry. Each entry: `{speaker: css}` by name (unknown speakers ignored) or a list of css strings in `Annotation.labels()` order. A bare string raises. Default: sorted union of all labels → matplotlib Set1 (pyannote's palette), deterministic for a label set.

## 8. Spectrogram

`colormap=` any matplotlib name (validated). `fscale="mel"` (default) or `"linear"`; `fmax=` top of the axis in Hz (default Nyquist; 5000 is a good speech setting). numpy STFT: n_fft 512, hop 128, Hann, magnitude normalized by window gain, frames max-pooled into at most 4000 columns, computed in 1 M-sample chunks (bounded memory, every sample observed). dB range 80 below `max(peak, −60 dBFS)` so silence is dark. Frequency axis: mel-spaced rows (low band gets most of the height, each row = max over its bins) or linear (max-pooled when shrinking, bilinear when growing); high frequencies on top; column i centred at `(i + 0.5) / cols`. Encoded as JPEG q85 data URL, one per audio; the browser stretches it with the zoom. Needs matplotlib (+Pillow); without it the fft button shows nothing.

## 9. Layout and controls (browser)

Rows top to bottom:
1. Panel row 1: time readout (seconds, 3 decimals) | `◀` (−step) | play/pause (green circle) | `▶` (+step). Tooltips show the step.
2. Panel row 2: `none`/`all` (left, dark pill, toggles every audio; label = next action) | `FFT`/`WAVE` (small caps pill).
3. Gutter column 0: per-audio checkbox + name at each audio row (green check = playing, empty box = muted; press-hold-drag the pill to reorder the audio rows, `order` trait = permutation of the original indices, waveforms / spectrograms / mute white-outs / bound functions follow); at each lane group a `▸` / `▾` expander and the annotation name (click = mask cycle; press-hold-drag beyond 5 px = move the group up or down, a blue line shows the drop position, the new order is written to `groups`, mask keys follow). Width = widest entry or the all/none button (dark filled pill).
4. Gutter column 1: one coloured speaker label per lane row, only for the annotations whose expander is open (`expanded` trait / argument); collapses when none is open (click = mask cycle). Width = widest label. Light colours get a dark halo.
5. Audio rows, `height` px each: waveform (first paint from 2000 peaks, exact samples after background decode) or spectrogram image (`fft`). Function lines overlay (SVG, 1 px). While two or more audios are **selected** a last **mix row** (blue frame, label `mix`, no checkbox) shows the mean of the selected audios, i.e. what is played; it appears and disappears with the selection and re-renders on every toggle (waveform in the browser, spectrogram recomputed by Python when `enabled` changes).
6. Timeline, 16 px: ticks ≥ 6 px apart, secondary labels ≥ 40 px, primary ≥ 90 px, decimals for sub-second steps; recomputed on every zoom.
7. Lane groups, `row_height` px per label row, in the order of `groups` (draggable); segments are plain coloured bars (speaker name only in the `▸` column and in the hover title).

Everything inside the scroll wrapper is sized in percent of the duration: zoom is CSS stretching, no recompute. The red playhead spans all rows.

Interactions:
- Wheel over the audio/lanes: zoom around the pointer (0.25 scale per notch, max 2000 px/s); auto-scroll is paused for 400 ms around wheel events so playback does not fight the zoom. Horizontal scroll pans.
- Click anywhere in the rows: seek and play. Space: play/pause. ← / →: nudge by `step`. Keys act only when the widget has focus (any click inside focuses it) and never reach the document.
- `☑` toggles: mute via per-audio WebAudio gain nodes (10 ms ramp), playback continues.
- Click on a name cycles three states: neutral → keep its segments (solid red outline) → keep the complement, the unlabelled parts (dashed red outline) → neutral. An annotation name means the union of all its speakers; a legend entry one speaker. Several active names intersect. Playback keeps only the result: outside it the audio is silenced (gain automation, 5 ms ramps) and the playhead skips (jumps to the next kept segment, pauses at the end of the last). The view keeps its time axis; excluded parts are whited out at 92 %. Python: `solo=["ref", "!ref/speaker91"]`, `w.solo_(...)`, trait `solo` holds the keys with a `!` prefix for the complement. Dimming (`w.dim(...)`) is Python-only.
- A muted audio row (checkbox off) is whited out the same way, like a cut.
- Pause writes the exact position to the readout and to Python.

## 10. Python ↔ browser

Traits (synced): `audio_url, duration, functions, peaks, spectrograms, audio_names, channels, weighted, height, row_height, view, step, zoom, autoplay` (Python → browser) and `groups` (order changes by drag), `order` (audio row order), `expanded, current_time, playing, enabled, solo, dimmed` (both ways, `echo_update=False` so a slow link never replays stale values).

Methods: `w.seek(t)`, `w.play()`, `w.pause()`, `w.solo_(*items)` (`"ref"`, `"ref/speaker90"`, `("ref", "speaker90")`; no args clears), `w.dim(*items)` (same keys; no args clears), `w.enabled = [...]`, `w.expanded = ["ref"]`, `w.groups` (current order).

Arguments for initial state: `enabled=[bools]`, `solo="ref"` or list, `dim=[...]`, `zoom=px_per_second` (0 = fit), `time=seconds`, `autoplay=True` (browsers may block until a click), `view="fft"`, `legend="side"|"both"`, `step=seconds`.

Validation: `view` is an enum; `height`, `row_height` > 0; `step` > 0; `zoom`, `time` finite ≥ 0; `enabled` length = number of audios; unknown annotation or speaker in `solo` / `dim` raises.

## 10b. Standalone export, Comet, Weights & Biases

- `w.html(title="wavlens")` → one self-contained HTML string: the IIFE bundle (`wavlens/wavlens.iife.js`), every trait as JSON, an in-page model. Audio, spectrograms, lanes, functions, zoom, keys, mute, solo, dim all work; only the Python sync is absent. `w.save_html(path)` writes it.
- `w.log_comet(experiment, name="wavlens", step=None)`: `experiment.log_html(...)` (HTML tab) plus `log_asset_data(...)` as `<name>_<step>.html`.
- `w.log_wandb(key="wavlens", step=None)`: `wandb.log({key: wandb.Html(...)})`, renders in the run page.
- Pattern in a training loop: `wavlens.show(file, {"hyp": model.diarize(file), "ref": ref}).log_comet(exp, step=epoch)`.
- Size = widget state (≈ 1.3 MB per 30 s mono at 16 kHz) + 64 KB bundle.

## 11. Caching and limits

- Content-hash LRU (`blake2b` of the samples, 256 MB total): WAV data URL, peaks, spectrogram per (audio, colormap, height). Same audio shown again in another cell costs nothing.
- Browser decodes the WAV once (WebAudio); the decoded buffer is reused for exact waveforms and solo masks.
- State size warning above 20 MB (≈ 10 min mono at 16 kHz). Very long audio works but bloats the notebook file.
- Measured: 30 s clip `show()` ≈ 30 ms, 10 min ≈ 110 ms, first paint in the browser ≈ 70–80 ms, play/arrow latency < 1 ms.

## 12. Files and tooling

`wavlens/__init__.py` (library), `js/wavlens.js` → `npm run build` → `wavlens/wavlens.js` (bundle, wavesurfer 7 core + Timeline + Zoom, no network). `tests/test_weights.py` (numeric, 27 checks), `tests/run_harness.py` + `harness.html` (Playwright, 15 browser scenarios, screenshots in `tests/out/`). `examples/wavlens.ipynb` (one cell per form). After a rebuild restart the kernel (anywidget reads the bundle at import). `pip install -e ".[all]"`, `python -m build` for PyPI.
