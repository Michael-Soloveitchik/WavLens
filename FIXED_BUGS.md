# wavlens fixed bugs

Findings from the 2026-10-13 bug hunt (Python fuzzing, numerical audit, browser harness), each with its fix. Status: ✅ fixed and covered by `tests/test_weights.py` or the harness, 📝 documented convention, ⏳ open.

## Data integrity

| # | Finding | Fix | Status |
|---|---|---|---|
| 1 | One NaN sample made the global gain NaN: every channel full-scale DC, peaks NaN, widget failed to serialize; inf zeroed everything; order-dependent | `_np` rejects NaN/inf in audio, functions and weights with a counted `ValueError` | ✅ |
| 2 | `(path, sr)` overrode the file's rate without resampling (half-speed playback, 60 s duration) | `(path, sr)` and `{"audio", "sample_rate"}` resample the file to `sr` | ✅ |
| 3 | Spectrogram dropped audio beyond 128 s @16 kHz (hop grew past the window; 3.6 % observed at 1 h) | hop fixed at n_fft/4, frames max-pooled into ≤ 4000 columns, chunked | ✅ |
| 4 | All-silent audio rendered as the brightest colour; −140 dBFS noise fully textured | dB reference = `max(peak, −60 dBFS)`, magnitude normalized by window gain | ✅ |
| 5 | NaN in a function → flat 0.5 line; a 3 ms pulse vanished (point sampling) | NaN raises; downsampling max-pools | ✅ |
| 6 | `colors="red"` zipped characters (`"r"`, `"e"`) as colours | string entries raise; per-annotation dict/list only | ✅ |
| 7 | Labels `1` and `"1"` collided as rows; 2-D function arrays flattened into a zigzag | collisions after `str()` raise; 2-D → one curve per column | ✅ |
| 8 | Cubic overshoot: negative gains in weights, rescaled plots | cubic clipped to the data range | ✅ |
| 9 | Resampling rounding (44.1 kHz → 16 kHz one sample longer) rejected matching recordings | up to `max(2, sr/1000)` samples trimmed | ✅ |
| 10 | `dim="sp"` iterated as characters; int names and names containing `/` unreachable in `solo`/`dim` | strings accepted, names `str()`-ed, tuple form `("ann", "speaker")` for slashes, speakers validated | ✅ |
| 11 | Zero-length / 0-d / 3-D audio, Python lists, bf16 tensors, `sample_rate` 0 / negative / `"16k"`, `(path, path)`, `{"waveform", "sample_rate": None}` crashed late with cryptic errors | validated in `_np`, `_audio`, `_check_rate` with clear messages; `(path, path)` is two audios | ✅ |
| 12 | `height=96.0`, 0, unknown colormap, empty function, `weights=[None, 1]`, cubic on 1 point → PIL / KeyError / numpy errors | traitlets validators, eager colormap check, explicit checks | ✅ |
| 13 | `interpolation="bogus"` passed when nothing was resampled | validated at the top of `show()` and in `_resample` | ✅ |
| 14 | int16 audio not scaled: mixed with float audio the float row was flat | integer dtypes divided by their max | ✅ |
| 15 | Segments outside the audio or negative passed through | clipped to `[0, duration]` | ✅ |
| 16 | `view="FFT"`, `legend="bogus"`, `step=0`, `zoom=-5`, `time=nan`, `enabled="ab"`, `solo="sp/nobody"` reached the browser | enums and validators on the traits; `enabled` length checked; speakers checked | ✅ |
| 17 | Positional `Audio_0` vs dict key `Audio_0`, `sp_0` vs named `sp_0` collided | `_unique` adds `~1`, `~2` | ✅ |
| 18 | `{"audio": path, "sample_rate": 8000}` ignored the rate; both `audio` and `waveform` keys silently picked one | rate applied as target; both keys raise | ✅ |
| 19 | SlidingWindowFeature longer than the audio drew past the right edge | frames with middle > duration dropped | ✅ |
| 20 | No size cap; 10 min @48 kHz = 77 MB state silently stored in the notebook | warning above 20 MB; cache avoids recomputation | ✅ (file-URL serving open ⏳) |

## Second pass (re-audit of the rewrite)

| # | Finding | Fix | Status |
|---|---|---|---|
| 24 | File audio bypassed the NaN guard (a float WAV with one NaN still produced the full-scale thump) | `_read_audio` output goes through `_np` | ✅ |
| 25 | 2-D function arrays with values outside [0, 1] were sigmoid-squashed; a non-probability frame matrix binarized into one segment | sigmoid only for `.logits`; a raw matrix annotation must be in [0, 1] or it raises | ✅ |
| 26 | Any string was an RTTM annotation (empty group, no error); a single-line RTTM was read as an audio path | strings must start with `SPEAKER ` (newline not required), else `TypeError` | ✅ |
| 27 | Cached peaks list shared between widgets (mutating one changed the next) | cache hands out copies of lists | ✅ |
| 28 | Empty or fully out-of-range SlidingWindowFeature, 3-D data → numpy errors | explicit `ValueError`s | ✅ |
| 29 | Any list of multi-word strings (paths with spaces) detected as `start end label` lines; records without `end` → `float(None)` | lines need two numeric tokens; records need start and end | ✅ |
| 30 | `threshold="0.5"` / NaN, `height=96.0` / 0 reached numpy or PIL first | validated at the top of `show()` | ✅ |
| 31 | `solo=3` star-unpacked; `enabled="ab"` / dict, `autoplay="no"`, `weights="0.5"`, `time` past the end accepted | int solo accepted; the rest raise | ✅ |
| 32 | `_unique` renamed the user's explicit `Audio_1` instead of the auto name | explicit names win, auto names yield (`Audio_1_`) | ✅ |
| 33 | `(x, "16000")` reported an unrecognized tuple | names the bad rate | ✅ |
| 34 | Max-pooling dropped negative excursions (dips, mute gaps) sharing a bin with a larger value | pooling keeps the sample with the largest \|value\|, sign kept, for functions and SlidingWindowFeature | ✅ |
| 35 | SlidingWindowFeature subsampled `[::step]` instead of pooling (pulses at odd frames vanished past 4000 frames) | pooled like the raw path, time = bin mean | ✅ |
| 36 | Spectrogram columns a half hop late (+4 ms at 16 kHz) | padding shifted by hop/2; impulse lands on its column centre | ✅ |
| 37 | Pooled raw functions drew the last partial bin on a uniform grid (up to half a point early, 0.9 s at 1 h) | bin-mean time for the raw path, same as SlidingWindowFeature; raw and SWF now agree exactly | ✅ |
| 38 | `_resample(y, 0)` raised `ZeroDivisionError`; a 3-tuple solo/dim key dropped its third item | both raise `ValueError` | ✅ |
| B21 | `weighted` set from Python not applied until the next toggle | `change:weighted` → `applyEnabled` | ✅ |
| B22 | Python seeks smaller than 0.15 s ignored (guard against our own echo) | own updates flagged; every Python seek applied | ✅ |
| B23 | Ruler primary labels (0.05 s) not a multiple of secondary (0.02 s): collisions at max zoom | primary interval chosen as a multiple of the secondary | ✅ |
| B24 | Segments had a 6 px minimum width from padding | `text-indent` instead of padding | ✅ |
| B25 | Wheel over the gutter armed the auto-scroll hold | wheel listener only on the audio host | ✅ |
| B26 | Human double-click on the waveform re-seeked and restarted the source (first 140 ms heard twice) | a second click within 300 ms does not restart | ✅ |

Documented, not changed: `nearest` and `previous` coincide under the frame-centre convention except at exact ties; a SlidingWindowFeature draws at pyannote's window middles, a raw array at step-tiled centres (differ by (window − step)/2); a dim/solo tuple means (annotation, speaker); keys do nothing when the widget is not focused; zoom-out below fit re-renders (40–67 ms) without visible change; a viewport shrink while zoomed can leave the cursor off-screen until the next seek.

## Time conventions

| # | Finding | Fix | Status |
|---|---|---|---|
| 21 | Three conventions coexisted (raw functions at frame start stretched to the end, SlidingWindowFeature at middles, spectrogram columns half a hop off), up to 12 ms apart; function x rounded to 1e-4 (180 ms at 1 h) | one rule: sample k of L at `(k + 0.5) / L`; spectrogram columns centred the same way; x rounded to 1e-6 | ✅ 📝 |
| 22 | `"previous"` was nearest-below on a linspace grid; downsampling skipped frames; `"nearest"` banker's rounding on ties | `previous` = frame hold, `nearest` = round-half-even of `t − 0.5` (documented), downsampling pools | ✅ 📝 |
| 23 | Unequal weights cost 20 dB SNR per decade in 16-bit PCM | documented; float WAV would double the payload | 📝 |

## Browser

| # | Finding | Fix | Status |
|---|---|---|---|
| B1 | `minPxPerSec: 1` made wheel zoom scroll to the end and waste the first notch | option removed | ✅ |
| B2 | Legend toggle rebuilt the gutter, resized the wrapper, re-rendered the waveform (visible jump) | columns sized once per draw; legend toggle only flips visibility/text | ✅ |
| B3 | ipywidgets echoed stale `current_time` / `playing` over a slow link: seek-backs, double play toggles | `echo_update=False` on browser-driven traits | ✅ |
| B4 | Readout stale by one frame after pause | pause handler writes `getCurrentTime()` to readout and Python | ✅ |
| B5 | Side-legend label invisible on a same-colour segment; Set1 yellow invisible on white | labels moved to their own column; dark halo on light colours | ✅ |
| B6 | Per-channel `normalize` made the first paint differ from the decoded render (jump on decode) | one global gain in Python, `normalize` off | ✅ |
| B7 | Spectrogram plugin recomputed the FFT per wheel step (lag, snap) | Python-rendered image, CSS stretch | ✅ |
| B8 | Zooming while playing was dragged back by auto-scroll | auto-scroll paused 400 ms around wheel events | ✅ |
| B9 | SVG `className` assignment threw (read-only on SVG) | `setAttribute("class")` | ✅ |
| B10 | Column labels could land on the timeline when host CSS changed plugin heights | positions taken from `lanes.offsetTop` | ✅ |
| B11 | Single click before a double-click flashed dim | dim delayed 250 ms, cancelled by double-click | ✅ |
| B12 | Play before the audio was decoded: cursor ran from 0 with no sound | play deferred until `canplay` when the buffer is missing | ✅ |
| B13 | Cursor one frame behind the clock after pause; first ◀ moved it right | pause re-renders the cursor at the exact time | ✅ |
| B14 | `current_time` stale after several nudges in one tick (10 Hz throttle, no trailing send) | trailing flush 120 ms after the last update | ✅ |
| B15 | Zoom anchor jumped after a pan/seek with the mouse still (upstream Zoom plugin caches the pointer time) | plugin cache invalidated on `scroll` and `seeking` | ✅ |
| B16 | Solo/dim keys survived a `groups` change that removed their label: full white-out, silent playback, no way out | keys dropped when their annotation/speaker disappears | ✅ |
| B17 | Out-of-range or NaN `current_time` from Python showed raw values and a cursor at 150 % | clamped to `[0, duration]`, NaN ignored | ✅ |
| B18 | End of file sent `current_time` slightly above the duration | clamped | ✅ |
| B19 | Short or empty `enabled` from Python broke toggles and gains | padded/truncated in the browser, length validated in Python | ✅ |
| B20 | Rendered while hidden (collapsed output): gutter measured 0 px, names drawn over the waveform | ResizeObserver re-draws when the widget first becomes visible | ✅ |

## Rule changes requested after the audits

| # | Change | Status |
|---|---|---|
| R1 | A bare array audio is rejected: every audio carries its rate (`(x, sr)`, `{"waveform", "sample_rate"}`, file). `sample_rate=` is only an optional target. Mixed rates without a target resample to the highest rate so one time line holds | ✅ |
| R2 | Per-audio toggle restyled (SVG checkbox: green check = playing, empty box = muted); all/none is a dark filled pill; `FFT`/`WAVE` small-caps pill | ✅ |
| R3 | No double-click: a single click on a name cycles neutral → keep → complement → neutral; annotation = union of its speakers; several active names intersect; `!` prefix on the `solo` keys for the complement | ✅ |
| R4 | Spectrogram looked half blank (speech has nothing above 4 kHz on a linear axis): mel frequency axis by default, `fscale="linear"` keeps the old axis | ✅ |
| R5 | A muted audio row is whited out like a masked gap | ✅ |
| R6 | `fmax=` caps the spectrogram frequency axis (mel or linear), e.g. 5000 Hz for speech, so the empty top band is not drawn; image rows always equal `height` | ✅ |
| R8 | Legend button and `legend=` removed: a `▸`/`▾` expander beside each annotation name opens that annotation's speaker column (`expanded=` argument / trait); the column collapses when none is open | ✅ |
| R9 | Annotation groups are reorderable by dragging the name (5 px threshold, blue drop line); order persists in `groups`, mask keys remapped | ✅ |
| R10 | Mix row: with ≥ 2 audios a framed, non-selectable row above the timeline shows the mean of the selected audios (waveform and spectrogram), live on toggles | ✅ |
| R11 | Playback gain is 1/m over the m selected audios, explicit weights included (was 1 with weights) | ✅ |
| B27 | Mix-row frame appended on every redraw (DOM leak, background darkening) | class `mixframe`, cleared in `draw()` | ✅ |
| R12 | Audio rows are reorderable by dragging the checkbox pill; `order` trait / argument (permutation, validated) | ✅ |
| R13 | Mix row shown only while at least two audios are selected; rows, heights and toggles follow live | ✅ |
| R14 | No text on the segments; speaker names only in the expander column (and the hover title) | ✅ |
| B28 | Column entries placed one audio row off after the mix row appeared/disappeared (lanes measured before wavesurfer re-rendered) | columns re-placed after the re-render when the lanes moved | ✅ |
| B29 | Drag remap deleted solo/dim keys when a remapped key collided with another active key (a mask silently changed) | remap built from snapshots into fresh sets | ✅ |
| B30 | A no-op drag released off the label left the `dragged` flag set and swallowed the next real click | flag set only after a real reorder, cleared on pointerdown | ✅ |
| B31 | Drag listeners survived pointer leaving the window; no pointerId filter, no pointercancel | pointerId filtered, pointercancel and window blur end the drag | ✅ |
| B32 | `order` from the browser side unchecked for range (crash in `withMix`) | indices range-checked | ✅ |
| B33 | Unknown names in `expanded` written back; drag marker shown when a single row cannot move | unknown names dropped on save; no drag with fewer than two rows | ✅ |
| B34 | With a mask active, no sound after the playhead skipped a gap: the mask automation was computed once at solo time (the `play` / `seeking` re-scheduling hooks had been lost in a rewrite) and, when re-armed from the `seeking` event, used the pre-jump position | re-scheduled on every play and seek (after the player takes the new position) and explicitly from the jump target | ✅ |
| R15 | Clicking the rows started playback; now click = seek only, paused stays paused | ✅ |
| R16 | Calmer controls: 9 px hollow checkbox with a thin green check, no unicode glyphs; expander is a small SVG chevron (`button.exp`, rotates when open) | ✅ |
| R7 | Renamed waviz → WavLens (`wavlens` package, `WavLens` class, `wavlens.js`, `wavlens.iife.js`), repo on branch `main`, README with screenshots from `docs/`, `.github/workflows/publish.yml` for PyPI releases | ✅ |

## Third pass (expander, drag, mix row, mel axis)

| # | Finding | Fix | Status |
|---|---|---|---|
| 39 | Mel axis with `fmax` below Nyquist mis-scaled frequencies (440 Hz read 717 Hz at `fmax=5000`, tones above ~3.1 kHz vanished): bins were sliced before the mel mapping | mapping uses the full-band bin count | ✅ |
| 40 | `expanded=False` / `None` raised `TypeError` | falsy → none; names validated against the annotations, duplicates dropped | ✅ |
| 41 | Reassigning `w.groups` from Python left solo/dimmed/expanded keys pointing at the wrong annotation | observer remaps keys by name, drops vanished ones; `groups` shape validated | ✅ |
| 42 | Mix spectrogram computed twice per `show()` (two cache key forms) | one key form | ✅ |
| 43 | `fmax="5000"` / `True` accepted through `float()` | numbers only | ✅ |
| 44 | `_mix` kept the audio copies for the kernel lifetime | released in `close()` | ✅ |
| 45 | Positional list / dict of audios or annotations rejected despite the docs | unpacked when all items are audios or all annotations | ✅ |
| 46 | Dependencies: matplotlib, Pillow, scipy, ipywidgets, traitlets were optional or implicit | all runtime deps explicit in `pyproject.toml` and `requirements.txt`; verified by installing the wheel alone in a fresh venv | ✅ |

## Lifecycle and sync

| # | Finding | Fix | Status |
|---|---|---|---|
| L1 | Destroying a view (clear output, re-render) never stopped the audio nor closed the AudioContext: zombie playback to the end of the buffer, model stuck at `playing=true` | cleanup pauses, destroys the player, closes the context | ✅ |
| L2 | Every destroyed view leaked whole (DOM, wrapper, decoded buffer, running context) | listeners removed, observer disconnected, timers cleared, context closed | ✅ |
| L3 | One AudioContext per widget, never released; 41 widgets = 41 running contexts | released on destroy; live widgets still own one each (Chromium 145 showed no cap) | ✅ |
| L4 | Two views of one model both played: doubled audio | the view that started playback owns the sound; others follow `current_time` silently | ✅ |
| L5 | `canplay` listener and wheel timer outlived the view | cleared in the cleanup | ✅ |
| L6 | Python seeks < 0.15 s ignored; out-of-range readouts | fixed earlier (B22, B17) | ✅ |
| L7 | `w.enabled = list(np.array([...]))` stored `np.bool_`, which the kernel cannot JSON-serialize | validator casts to `bool` | ✅ |
| L8 | Solo toggles copied the whole decoded buffer (106 MB, 300 ms stall at 5 min stereo) | gain automation from the current position, no allocation | ✅ |
| L9 | `ANYWIDGET_HMR=1` needs `watchfiles`, not installed; new widgets kept the old bundle | documented: restart the kernel after `npm run build` | 📝 |
| L10 | Cleanup closed the AudioContext twice ("Cannot close a closed AudioContext" unhandled rejection) | close only when not already closed, rejection swallowed | ✅ |
| L11 | Initial `solo=` from Python not applied after the automation refactor (found by the standalone-HTML check) | `applySolo()` on ready | ✅ |
| L12 | Standalone export dumped no data: `class_own_traits` on the instance sees only anywidget's traits | declared class used | ✅ |

Verified by the audit: `echo_update=False` suppresses the echo for browser-driven traits; state up to 77 MB (15 min × 2) still renders (0.9 s in the harness), real JupyterLab check pending; comm messages are not rate-limited.

## Spectrogram

| # | Finding | Fix | Status |
|---|---|---|---|
| S1 | Bilinear downscale faded a 1-bin tone to 0.25 at `height=20` (0.71 at 96) | frequency axis shrunk by max-pooling, bilinear only when growing; tone stays at full value at 20/40/96/300 | ✅ |
| S2 | `fft` button toggled to a blank block when matplotlib is missing | button disabled with a tooltip | ✅ |
| S3 | Black function line invisible over magma | white halo under every function line | ✅ |
| S4 | Segments past the audio end inflated the wrapper and the ruler labels | segments clipped in Python | ✅ |
| S5 | `max_cols` off by one (4001 columns) | `cols = min(max_cols, frames)` | ✅ |

## Open (from the feature scout and pending audits)

- Large audio should be served through a file URL instead of a base64 data URL.
- Reports from the spectrogram, browser-abuse and lifecycle hunts are pending; items will be appended here.
