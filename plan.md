# wavlens — plan and blueprint

How wavlens was built, decided and verified, in enough detail to recreate it from scratch. Reference docs: `FUNCTIONALITIES.md` (what it does), `FIXED_BUGS.md` (what went wrong and how it was fixed), `README.md` (install, build, test).

## 1. Goal

Visualize a diarization hypothesis (pyannote, NeMo, WavLM, any RTTM-like format) on top of the audio, interactively in Jupyter: waveform or spectrogram rows, a time ruler, one lane group per annotation with one row per speaker, 0..1 function overlays, a red playhead that moves with playback, zoom with the wheel, click to seek, per-audio mute/weights mixing, solo per speaker or annotation with gap skipping, everything also settable from Python arguments. Minimal, flat, pyannote-native, pip-installable, no network at runtime.

## 2. Architecture

```
Python  wavlens/__init__.py  (one module, one class)
        show(...) -> validates inputs -> loads/resamples audio -> applies weights
                  -> WAV (16-bit, global peak) + peaks (2000 abs-max bins) + spectrogram JPEG (chunked STFT, max-pooled)
                  -> groups (lanes) + functions (2000 pts, frame-centre time) -> WavLens traits
        WavLens(anywidget.AnyWidget): synced traits, validators, seek/play/pause/solo_/dim
        content-hash LRU cache (256 MB): WAV, peaks, spectrogram per (audio, colormap, height)

Browser js/wavlens.js -> esbuild -> wavlens/wavlens.js  (wavesurfer.js 7 core + Timeline + Zoom bundled)
        panel rows (transport, all/none, fft, legend) | gutter col0 (☑ per audio, annotation names)
        | gutter col1 (speaker legend, collapsible) | wavesurfer scroll wrapper:
          audio rows (waveform canvases | spectrogram <img> per audio), overlay (function SVGs, gap white-outs),
          timeline host (plugin), lanes (segment divs in percent of duration)
        WebAudio backend: player -> maskGain (solo automation) -> ChannelSplitter -> gain_i -> destination
        percent sizing = zoom is CSS stretching; cursor spans all rows; column entries placed from lanes.offsetTop
```

Why these choices (each one was tested against the alternative):
- anywidget over Gradio/gryannote: works in Jupyter/VS Code, no server, gryannote pins gradio 4.27.
- wavesurfer bundled over CDN: instant load, offline; esbuild one-liner.
- peaks first, decode later: first paint ≈ 70 ms; exact samples swapped in on `canplay`.
- spectrogram in Python as an image over the wavesurfer Spectrogram plugin: the plugin recomputed the FFT per wheel step (lag + snap); an image stretches instantly.
- WebAudio backend over MediaElement: per-audio gains, mask automation, no multichannel playback limits.
- solo = silence + skip, view unchanged (the cut-and-concatenate variant was built and rejected: the time axis must stay).
- legend column collapses (an always-present hidden column was built and rejected).
- `echo_update=False` on browser-driven traits: the ipywidgets echo over a slow link replayed stale times.

## 3. Recreate from scratch

1. Environment: Python ≥ 3.10 with `anywidget numpy soundfile pyannote.core matplotlib scipy`; Node ≥ 18; for tests `playwright` + chromium, `jupyter`, `pyannote.database`, `torch`. Do not call `torchaudio.load` or `pyannote.audio.Audio` in this conda env (segfault); use soundfile.
2. Files: `pyproject.toml` (setuptools, package-data `wavlens/wavlens.js`, extras spectrogram/cubic/all/dev), `package.json` (`esbuild js/wavlens.js --bundle --format=esm --minify --outfile=wavlens/wavlens.js`, deps esbuild + wavesurfer.js@7), `.gitignore` (node_modules, dist, tests/out, tests/fixture*.json, __pycache__), `LICENSE` MIT.
3. Python module, in this order: array helpers (`_np` finite/int-scaling), type predicates (`_is_audio`, `_is_ann`, `_is_fn`, format detectors), `_named`/`_unique`, cache, audio loading (`_read_audio`, `_audio` returning carried rate and target, `_to_rate`, `_load_audios` with parallel reads and length trimming), `_peaks`, annotation parsers (records, lines, RTTM/UEM, files, `_matrix`, `_annotation` with clipping), `_color_specs`/`_groups`, `_resample` (frame-centre, `_pool` by largest |value|, `_bin_mean`), `_weight`/`_match`, `_bind_functions`/`_functions`, `_spectrogram` (chunked, pooled, −60 dBFS floor, pad `n_fft/2 − hop/2`), `WavLens` traits + validators + methods, `show()` with validation at the top.
4. JS module: helpers (`el`, `svg`, `round`, `pill`, `text`, `widest`, `ticks`, `union`); `render()` builds panel, gutter, wavesurfer (peaks + duration, backend WebAudio, Zoom plugin registered after creation with cache invalidation on scroll/seek), hosts appended to the wrapper (spec, overlay, timeline host, lanes), Timeline plugin with our `redraw` listener registered first, gain graph, `applyEnabled`, solo (`segsOf`, `scheduleMask`, `drawMask`, `applySolo`), dim with 250 ms delay, `draw()` (toggles, spectrogram imgs, function SVGs with halo, lanes, column entries at measured tops, column widths), controls (view, legend, nudge, keys, wheel hold on the audio host, initial state on ready), sync (trailing flush, own-update flag, clamps, pause re-render, active-view guard), cleanup.
5. Build: `npm install && npm run build`. Restart the kernel after every rebuild.
6. Tests: `tests/test_weights.py` (numeric reference implementations for weights, interpolation, functions, forms); `tests/make_fixture.py` dumps every synced trait of a real `show()` call; `tests/harness.html` fakes the anywidget model and loads the bundle; `tests/run_harness.py` drives Chromium through the scenarios (load timing, zoom anchoring, fft, ticks, legend, dim, solo, transport, keys, row heights, fixed gutter, decode swap, playback stability, latency, column geometry) and screenshots `tests/out/`.
7. Notebook `examples/wavlens.ipynb`: one `show()` per cell for every input form; execute headless with `--ExecutePreprocessor.store_widget_state=False`, then `--clear-output` (stale widget outputs otherwise bloat the file and fail to render).

## 4. Verification workflow used

Parallel agents, each report-only, each re-run after fixes until clean:
- harness (Playwright scenarios, extended per feature),
- notebook (examples, executed headless),
- numeric verification of weights/interpolation (27 checks),
- five adversarial hunts: Python input fuzzing (~270 cases), interpolation/alignment math (ms errors per convention), spectrogram rendering (tone/chirp/impulse/silence probes, deep zoom), browser abuse (timing, zoom, solo races, resize, alignment, 2000 segments), lifecycle/sync (destroy, many widgets, two views, echo, message size).
Fix-test-fix: fixes go in one batch, tests and harness re-run, agents re-dispatched on the new code, findings and fixes appended to `FIXED_BUGS.md`. Outcome: 38 data/convention items, 26 browser, 5 spectrogram, 9 lifecycle fixed; the only open item is serving very long audio by file URL.

## 5. Measured numbers (dev box, 2026-10-13)

| What | Value |
|---|---|
| `show()` 30 s / 10 min @16 kHz | 30 ms / 110 ms (cached repeat ≈ 0) |
| widget state 30 s mono / 10 min mono / 15 min stereo | 1.3 MB / 25 MB / 77 MB |
| first paint in the browser | 70–80 ms |
| play click / arrow key to readout | 0.7 ms / 0.2 ms |
| playback with 2000 segments + 240 polylines | 60 fps |
| spectrogram 1 h @16 kHz | 1.2 s, 350 MB peak on top of the input |
| overlay alignment (functions, spectrogram, lanes) vs waveform | ≤ 1.6 ms at 30 s and 61 s, exact at 1 h |
| solo toggle | no allocation (gain automation) |

## 6. Not built, possible later

- Serve long audio through a Jupyter file URL instead of a base64 data URL (state size).
- A/B loop + drag-select region; playback speed; hover readout (time, function value, speakers); minimap/overview strip; word-level transcript lane (WhisperX/CTM) with click-to-seek; reference-vs-hypothesis error lane coloured by DER component; per-segment confidence as opacity; F0/intensity curves with NaN gaps and a Hz axis; probability heatmap lanes; time-range crop + Python-synced view window; PNG / self-contained HTML export (also the path to Comet / W&B logging); segment editing + RTTM export.
- Static PNG / mp4 export with matplotlib + ffmpeg.

## 7. Library survey (2026-10-13)

| Candidate | Verdict |
|---|---|
| `pyannote.core.notebook` | palette reused (Set1); static only |
| `gryannote` (Gradio, wavesurfer) | design reference; pins gradio 4.27, not a dependency |
| `wavesurfer.js` 7 core + Timeline + Zoom | bundled; Spectrogram plugin rejected (per-wheel FFT) |
| `anywidget` | widget host |
| Voxmap-studio (2026 paper), Label Studio, peaks.js, Sonic Visualiser, Praat, Gecko, SeeVoice | feature ideas in section 6 |
| plotly/bokeh/dash, Gradio Audio, wandb/tensorboard audio | no cursor-synced audio; skipped |

## 8. pyannote / NeMo outputs covered

`DiarizeOutput.speaker_diarization` and `.exclusive_speaker_diarization` → `annotation=`; `serialize()` dict → `annotation=`; pipeline `hook` stages `segmentation`, `speaker_counting`, `discrete_diarization` (SlidingWindowFeature) → `function=`; Nemotron-3 Sortformer `diarize()` segments and `include_tensor_outputs` probabilities → `annotation=` / `function=`; WavLM / HF frame logits → both; `pyannote.metrics` error analysis → an `Annotation` of statuses with `colors=`; RTTM, UEM, LAB, SegLST, NeMo manifests as files. Not yet: speaker embeddings scatter, speech separation sources, word-level CTM.
