# WavLens: theory, comparison, specification

Minimal report. Three parallel surveys (viewers, rendering/playback theory, research features) on 2026-10-17, condensed. Claims about other tools are as surveyed at the cited URLs; WavLens facts are from `wavlens/__init__.py`, `js/wavlens.js`, `FUNCTIONALITIES.md`.

## 1. Model

**Data.** A *scene* = audios `a_i ∈ ℝ^T` at one rate `sr` (resampled to the highest), annotations `A_j : ℝ → 2^{labels}` (pyannote `Annotation`, one lane group each, one row per label), functions `f_k : [0, D] → [0, 1]` (min-max scaled), weights `w_i : [0, D] → ℝ`. Everything is referenced to one time axis `[0, D]`, `D = T / sr`.

**Time.** Positions are fractions of `D` (CSS percent). Zoom is a scale of the container, so no element is recomputed on zoom; only the ruler density and the spectrogram resampling are. Frame conventions: sample `k` of `L` sits at `(k + ½) / L · D` (frame centre). Segments are half-open `[s, e)`.

**Rendering.** Waveform: abs-max of `⌈T/2000⌉`-sample bins for first paint, exact samples after decode (wavesurfer 7). Spectrogram: Python STFT → dB → max-pool to `≤ 4000` columns → mel rows (max over bins) → colormap → JPEG; browser box-resamples when columns exceed pixels. Lanes: one `div` per segment. Functions: one SVG polyline (2000 points) per curve, `viewBox 0 0 1 1`, non-scaling stroke.

**Playback.** One `AudioBuffer` with the `N` audios as channels → mask gain `g_m(t)` → `ChannelSplitter` → per-audio gains `g_i = e_i / m` (`e_i` enabled, `m = Σ e_i`) → destination. Output `y(t) = g_m(t) · (1/m) Σ e_i a_i(t)`. The mix row draws exactly `(1/m) Σ e_i a_i`.

**Mask algebra.** Each clicked name `n` contributes a set `S_n ⊂ [0, D]` (speaker: its segments; annotation: union of its speakers) or its complement `S_n^c`. The kept set is `K = ⋂ S_n^{(c)}`. Playback: `g_m = 1_K` with 5 ms ramps, the playhead jumps from a gap to `min{s ∈ starts(K) : s > t}`, pauses at `max(K)`, and play from the end wraps to `min(K)`. `K = ∅` disables play. The time axis is never cut, so lanes of other annotations stay comparable.

## 2. Comparison with other tools

| Tool | Render | Annotation model | Playback | Jupyter | Has that WavLens lacks | Lacks that WavLens has |
|---|---|---|---|---|---|---|
| pyannote.core repr | matplotlib PNG | greedy row packing | none | native, static | nothing | audio, zoom, per-speaker rows, solo |
| pyannote.audio preview | matplotlib → mp4 (5 fps) | subplots | video | video | shareable mp4 | interactivity |
| gryannote | Svelte + forked wavesurfer, Regions | all speakers on one canvas | media element | needs Gradio server | editing, RTTM export, minimap | lanes, spectrogram, multi-audio, mix, solo |
| meeteval.viz | D3 + canvas | per speaker: ref band + hyp band, word boxes coloured by error | side audio server | yes (CDN) | word-level ref/hyp alignment, error colours, minimaps | waveform, offline audio, mute/solo |
| Label Studio Audio | own canvas, layers | regions on one track | WebAudio, speed, loop | no | region editing, ffmpeg decode, loop | lanes, formats, mix, notebook |
| Audacity | native | label tracks, auto sub-rows | full transport, loop, label nav | no | editing, unlimited zoom, loop | diarization formats, colours, Python |
| Praat / ELAN | native | one tier per speaker, no overlap in a tier | play selection, loop | no | real Hz/dB axes, hierarchy | overlap on a row, ref-vs-hyp |
| Sonic Visualiser | Qt panes + layers | layer taxonomy | transport, pane solo | no | Vamp analysis, sonification, Hz axes | diarization semantics, notebook |
| peaks.js / wavesurfer Regions | canvas / percent divs | segments on one layer | media element | no | precomputed peaks for hours of audio | lanes, spectrogram, mix |
| Prodigy, audio-annotator | wavesurfer | spans on one row | media element | no | correction workflow | everything research-side |
| Comet / W&B panels | player strip | none | player | n/a | nothing | WavLens is logged into them as HTML |

**Consensus and divergence.**
- Consensus WavLens follows: percent-of-duration positioning (wavesurfer Regions, gryannote); Hann STFT with 80 dB range (librosa, Audacity, Sonic Visualiser); perceptual colormaps; per-pixel max for waveform summaries.
- Divergence, advantage: one row per speaker with a collapse to one row (linguistics tools vs ML tools, WavLens offers both); Python spectrogram (numpy, any colormap, instant zoom); WebAudio gain graph (mix row and 1/m mixing exist nowhere else surveyed); solo = silence + skip with the axis preserved (others loop or play a region); ingestion breadth (RTTM, UEM, SegLST, NeMo manifests, HF logits, frame matrices in one call); the only interactive, offline, server-free notebook viewer, and the only one exporting to Comet/W&B.
- Divergence, risk: no editing (shared with pyannote repr and meeteval); functions are unit-less 0..1 (Praat, Sonic Visualiser keep Hz/dB); fixed 4000 columns blur at deep zoom on long files; no loop region, no word-level alignment view (meeteval's strength).

## 3. Theory per subsystem and verdicts

**Waveform.** Per-pixel abs-max is the exact vertical hull of the polyline in that column; decimation aliases, averaging shrinks a pulse of width `w` by `w/k`. Max composes across levels (max of maxes), which is why Audacity's 256/64k summary pyramid and BBC `audiowaveform` zoom levels work. wavesurfer 7 consumes abs-max per pixel, so the 2000-bin array is the right input. *Verdict: fine.* Optional: an inner RMS band (Audacity style) adds loudness at no cost since RMS composes via sums of squares.

**Spectrogram.** Hann: 4-bin main lobe, −31.5 dB first sidelobe (Harris 1978). Window ≈ 32 ms at any rate is narrowband, right for harmonics and speaker cues (Praat uses 30 ms for harmonics, 5 ms for formants). Hop `n_fft/4` meets COLA; since frames are max-pooled into columns, hop only matters at full zoom. Max over frames and over bins keeps the peak dB of tones and clicks exact; librosa's triangular mel sum would lose ~13 dB on a 1-bin tone in a 20-bin row. Price: the max of `N` Rayleigh noise bins sits 4–6 dB above their mean, so the noise floor brightens toward the top rows. The dB reference `max(peak, −60 dBFS)` with a window-gain normalisation gives physical dBFS and fixes librosa's failure on near-silent files. 8 bits over 80 dB = 0.31 dB per level. Magma/viridis are perceptually uniform (van der Walt & Smith 2015); jet is harmful (Borland & Taylor 2007). JPEG 4:4:4 keeps 1-px harmonics; 4:2:0 would blur them. *Verdicts: fine.* Two real improvements:
1. At 16 kHz, 512-point bins are 31 Hz while the lowest mel rows are ~18 Hz, so several rows clamp to one bin and show stairs. Zero-pad the FFT ×2 when rows are finer than bins (Audacity's default). Cost 2× FFT.
2. Browser resampling averages RGB after the colormap, which drifts mid-tones and dims a 1-column click by up to 12 dB in a 4:1 view. Ship the 8-bit dB plane (single-channel PNG, usually smaller than RGB JPEG), max-pool columns in JS, apply a 256-entry LUT in the browser. Also enables live colormap switching. Reassignment, multitaper, synchrosqueezing and CQT sharpen ridges but are non-additive under max-pooling; not needed here.

**Time alignment.** WavLens is self-consistent at frame centres; librosa/torch `center=True` place frame `t` at `t·hop`, a half-hop offset that does not matter as long as one convention is used. Model frames: pyannote segmentation 3.0 step 16.98 ms, receptive field 61.9 ms; wav2vec2/WavLM stride 320 samples, receptive field 400, true centre `320k + 200`; NeMo Sortformer 80 ms frames. Stretching a bare `(T, K)` matrix over the span skews by `(receptive field − stride)/2` plus up to half a frame: ≤ 12 ms for wav2vec-class, ≤ 40 ms for Sortformer. Invisible zoomed out, ~20 px at 2000 px/s. Half-open segments are correct for masks (adjacent segments union, zero-length intersections vanish) and match RTTM `start + duration`. *Improve:* accept `(array, step, offset)` or a `SlidingWindow` with bare arrays so frames are placed, not stretched.

**Playback.** `setTargetAtTime` is `v(t) = target + (v₀ − target)·e^{−(t−t₀)/τ}`; a-rate, sample-accurate within the 128-frame quantum. `τ = 1.67 ms` reaches 95 % at 5 ms and −60 dB at ~11.5 ms: the standard declick fade (Audacity, SoX). Automation is absolute in the audio clock, so re-arming on every play and seek is required, not optional (the bug of 2026-10-14 was exactly a lost re-arm). Gain masks beat buffer slicing: zero copy, no edited↔original time map, no second 230 MB/channel buffer for an hour at 16 kHz. Mixing: coherent sources (reference / degraded / enhanced of one utterance) sum with peak ∝ m, so `1/m` is the only clip-safe gain; equal-power `1/√m` suits uncorrelated sources only. LUFS (BS.1770-4, R128) solves a different problem, unequal loudness between sources before comparison. *Improve:* optional `normalize="lufs"` per audio before the `1/m`; subtract `ctx.outputLatency` (10–40 ms) from the readout; guard `N > 32` (ChannelSplitter limit).

**Interpolation.** Max-pool downsampling is the correct sup-norm display (mean shrinks pulses, median deletes them). For 0..1 curves the largest |value| is always the high side, so a brief dropout inside a high region disappears. *Improve:* pool to (min, max) pairs and draw a band, as for waveforms. Not-a-knot `CubicSpline` overshoots at steps; clipping trims the range but ringing remains. *Improve:* PCHIP (Fritsch & Carlson 1980) is shape-preserving and keeps probabilities in [0, 1] without clipping. `previous` = zero-order hold = pyannote step semantics; `nearest` with half-to-even ties is deterministic. *Verdict otherwise: fine.*

## 4. Proposed functionalities

**Summary (verified against `wavlens/__init__.py` and `js/wavlens.js` on 2026-10-17: none of these exists yet).**

| Priority | Features | State today |
|---|---|---|
| P0 | DER error lanes with collar (#1), mapped speaker colours (#2), posterior shading with a live threshold slider (#3), A/B keys (#4), loop region (#5), next/previous segment keys (#6) | absent; `threshold=` is a static argument, keys are only space / ← / → |
| P1 | speaker-count curve (#7), overlap lane (#8), word lane with tcpWER colours (#9), SI-SDR over time (#10), DNSMOS/STOI/PESQ curves (#11), difference row (#12), export mask as RTTM/UEM (#13), bookmarks (#14), UEM as scoring mask (#15), similarity-to-speaker curve (#16), per-speaker DER hover (#17), collar strips (#18), disagreement lane (#19), copy view as code (#20) | absent; a `.uem` file loads only as a plain lane (#15); `a − b` is reachable through `weights=[1, -1]` but not as a mix-row toggle (#12) |
| P2 | boundary match ticks (#21), hypothesis step slider (#22), embedding matrix / UMAP panel (#23), streaming latency lane (#24) | absent |

Grounded in pyannote.metrics, meeteval, NeMo, torchmetrics and the review tools above. Cost S/M/L, priority P0–P2.

| # | Feature | What | Compute | UI form | Cost | Pri |
|---|---|---|---|---|---|---|
| 1 | DER error lanes | derived annotation `missed / false_alarm / confusion` for hyp vs ref with collar and skip-overlap | `DiarizationErrorRate(collar, skip_overlap).optimal_mapping`, `rename_labels`, `discretize(0.01)`, frame compare, run-length | lane group `DER hyp/ref`, gutter shows `DER 12.3 %`, rows act as masks | M | P0 |
| 2 | Mapped colours | hyp speakers mapped to ref share the colour; unmatched grey | `optimal_mapping` (Hungarian) or `greedy_mapping` | `show(..., map_to="ref")`, Python only | S | P0 |
| 3 | Posterior shading + live threshold | frame posteriors as alpha strips on the lane, binarised at a slider | ship `(T, K)` probs; powerset via `Powerset.to_multilabel` | slider in panel row 2, `threshold` becomes a synced trait | M | P0 |
| 4 | A/B switch | keys `1..9` solo audio i, `Tab` cycles, position kept | nothing new (channels behind gains) | key bindings | S | P0 |
| 5 | Loop region | shift-drag on the ruler, `L` loops it or the kept segment under the playhead | browser only | ruler selection, `loop` trait | S | P0 |
| 6 | Next / previous segment | `N` / `P` jump along the active mask or error lane | reuse skip logic | keys, readout `3/17` | S | P0 |
| 7 | Speaker-count curve | active speakers per frame per annotation | `discretize().data.sum(1)` | function with `scale="integer"` | S | P1 |
| 8 | Overlap lane | overlapped speech per annotation | `Annotation.get_overlap()` | auto row, usable as mask | S | P1 |
| 9 | Word lane with tcpWER colours | WhisperX words coloured correct / sub / ins / del | `meeteval.wer.tcpwer` alignment | lane with text when zoom allows | M | P1 |
| 10 | SI-SDR over time | windowed SI-SDR of estimate vs reference stem | `torchmetrics` SI-SDR per 1 s window | curve bound to the estimate row | S | P1 |
| 11 | DNSMOS / STOI / PESQ curves | quality over time for enhancement | DNSMOS ONNX, torchmetrics | curves bound to the audio row | M | P1 |
| 12 | Difference row | mix mode `mean / diff` (`a − b`) | `weights=[1, −1]` exposed | panel toggle on the mix row | S | P1 |
| 13 | Export mask | current kept set or derived lane → RTTM / UEM | `Timeline.write_uem`, `Annotation.write_rttm` | `w.mask_timeline()`, save button in HTML export | S | P1 |
| 14 | Bookmarks | `B` drops a tick with a note; exported | trait `bookmarks=[(t, note)]` | ruler ticks | S | P1 |
| 15 | UEM as scoring mask | grey outside the UEM, feeds #1 | `DiarizationErrorRate(uem=)` | automatic for `.uem` rows | S | P1 |
| 16 | Similarity-to-speaker curve | cosine of each segment embedding to the clicked speaker's centroid | pyannote embeddings | curve appears on label click | M | P1 |
| 17 | Per-speaker DER hover | speech, missed, FA, confusion seconds per speaker | from #1 grouped by ref label | gutter tooltip | S | P1 |
| 18 | Collar strips | hatched ±collar around ref boundaries | ref boundaries ± collar | overlay toggled with #1 | S | P1 |
| 19 | Disagreement lane | frames where ≥ 2 mapped hypotheses disagree | frame compare | derived lane, usable as mask | S | P1 |
| 20 | Copy view as code | copies `wavlens.show(..., zoom=, time=, solo=, enabled=)` of the current state | traits | panel button | S | P1 |
| 21 | Boundary match ticks | hyp boundaries matched within tolerance green, else red | `SegmentationPrecision/Recall` | ticks on lane edges | S | P2 |
| 22 | Hypothesis step slider | `steps={epoch: hyp}` swaps the hyp group | Python packs annotations | slider in panel | M | P2 |
| 23 | Embedding matrix / UMAP panel | segment similarity, points seek | embeddings, `umap-learn` | side panel (leaves the time axis) | L | P2 |
| 24 | Streaming latency lane | chunk boundaries and emission delay for streaming Sortformer | per-chunk capture | ruler ticks + curve | M | P2 |

Rejected: click-to-edit boundaries (conflicts with click = seek; Label Studio and Prodigy own correction); a separate powerset display (redundant with #3); a Comet/W&B step slider inside the widget as P0 (both platforms already slide over per-step HTML).

## 5. Specification of the P0 set

**DER lanes (#1).** `show(..., der=("hyp", "ref"), collar=0.25, skip_overlap=False)`. Python: `m = DiarizationErrorRate(collar, skip_overlap).optimal_mapping(ref, hyp)`; `hyp' = hyp.rename_labels(m)`; discretise both at 10 ms over `[0, D]` (UEM if given); per frame: missed = ref speakers − hyp speakers, false alarm = hyp − ref, confusion = mapped mismatch; run-length each into segments; one lane group named `DER hyp/ref` with rows `missed`, `false_alarm`, `confusion`, fixed colours (orange, red, purple). The gutter name carries the DER value as text and the components as its title. Rows are ordinary rows, so clicking `confusion` keeps exactly the confusions and intersects with speakers. Collar strips (#18) are drawn from the same computation as a hatched overlay over the ref group.

**Mapped colours (#2).** `colors` resolution gains a step: when `map_to=<ref name>` is given, every other annotation's label `l` takes the colour of `m(l)` if mapped, else a grey from a fixed list. No browser change.

**Posterior shading (#3).** Frame matrices are kept as `uint8` probabilities per lane row (`(T, K)` trait `posteriors`, quantised to 8 bits, ≤ 1 MB per 10 min at 100 fps). The browser draws them as a canvas strip under the segments with alpha = probability; the slider writes `threshold`, the browser re-binarises and rebuilds the segments of that group; Python mirrors the trait so `w.threshold` and masks stay consistent. Placement uses `(step, offset)` when known (pyannote `SlidingWindow`, WavLM stride, Sortformer frame) rather than stretching.

**A/B, loop, next (#4–6).** Keys: digits solo one audio (others muted, position kept), `Tab` cycles, `L` toggles loop over the ruler selection or the kept segment under the playhead, `N`/`P` seek to the next/previous start of `K` (or of the error lane if one is active). Loop is implemented as a `timeupdate` guard (seek to the loop start at its end) and a `loop` trait `(t0, t1) | null`. All three are browser-only.

## 6. Sources

Viewers: pyannote.core notebook.py, pyannote.audio utils/preview.py, github.com/clement-pages/gryannote, github.com/fgnt/meeteval (viz/visualize.js), labelstud.io/tags/audio, manual.audacityteam.org (label tracks, spectrogram settings), fon.hum.uva.nl/praat/manual (TextGrid, Spectrogram settings), archive.mpi.nl/tla/elan, sonicvisualiser.org/doc/reference, github.com/bbc/peaks.js, github.com/bbc/audiowaveform, github.com/katspaugh/wavesurfer.js (regions.ts, renderer.ts), prodi.gy/docs/audio-video, github.com/CrowdCurio/audio-annotator, comet.com docs (log HTML/audio), docs.wandb.ai (media).
Theory: Harris 1978 (windows), Heinzel et al. 2002 (spectral estimation), O'Shaughnessy 1987 and Slaney 1998 (mel), Brown 1991 and Schörkhuber & Klapuri 2010 (CQT), Auger & Flandrin 1995 and Fulop & Fitz 2006 (reassignment), Thomson 1982 (multitaper), Daubechies et al. 2011 (synchrosqueezing), van der Walt & Smith 2015 and Borland & Taylor 2007 (colormaps), Mitchell & Netravali 1988 (resampling), W3C Web Audio API (AudioParam, ChannelSplitter, latency), web.dev "A Tale of Two Clocks", ITU-R BS.1770-4, EBU R128, ITU-R BS.1534-3, Fritsch & Carlson 1980 (PCHIP), Akima 1970, Plaquet & Bredin 2023 (powerset segmentation), Baevski et al. 2020 (wav2vec2), Chen et al. 2022 (WavLM), Park et al. 2024 (Sortformer).
Features: pyannote.github.io/pyannote-metrics, github.com/fgnt/meeteval, github.com/nryant/dscore, pypi.org/project/mdeval, docs.nvidia.com NeMo speaker diarization models, github.com/m-bain/whisperX, DNSMOS (github.com/microsoft/DNS-Challenge) and torchmetrics audio metrics.
