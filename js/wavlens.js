// Source for wavlens/wavlens.js. Build: `npm run build` (esbuild bundles wavesurfer in).
import WaveSurfer from "wavesurfer.js";
import Timeline from "wavesurfer.js/dist/plugins/timeline.esm.js";
import Zoom from "wavesurfer.js/dist/plugins/zoom.esm.js";

// Layout: two button rows, then a flex row of three columns: column 0 (☑ per audio,
// annotation names; as wide as its longest entry or the all/none button), column 1
// (per-row speaker legend; shown in side/both, collapsed in inline) and wavesurfer's
// scroll wrapper with the audio rows (waveform | spectrogram tiles), the timeline
// and the annotation lanes. Everything in the wrapper is sized in percent of the
// current duration, so zoom is CSS stretching and the red cursor crosses all rows.
// Column entries are placed from the measured DOM (lanes.offsetTop), never constants.
// Click a name (annotation = union of its speakers) to cycle: neutral -> keep its
// segments -> keep the complement -> neutral. Active names INTERSECT. Outside the
// result playback is silent and the playhead skips; the view keeps its time axis,
// gaps are whited out. Solid red outline = keep, dashed = complement. Muted audio
// rows are whited out the same way.
// The wrapper lives in a shadow root, hence inline styles everywhere.
const BAR_H = 24, TIMELINE_H = 16, FADE = 0.005;
const STEPS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
const ICON = {
  play: "M8 5v14l11-7z", pause: "M6 5h4v14H6zm8 0h4v14h-4z",
  back: "M6 6h2v12H6zm3.5 6l8.5 6V6z", fwd: "M16 6h2v12h-2zM6 18l8.5-6L6 6z",
};
const NS = "http://www.w3.org/2000/svg";

const el = (tag, css, text) => {
  const e = document.createElement(tag);
  e.style.cssText = css;
  if (text != null) e.textContent = text;
  return e;
};
const svg = (d, size) => `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="currentColor"><path d="${d}"/></svg>`;
const round = (d, size, bg, title) => {
  const b = el("button", `width:${size}px;height:${size}px;border:0;border-radius:50%;background:${bg};color:#fff;cursor:pointer;` +
    "display:flex;align-items:center;justify-content:center;padding:0;opacity:.92;box-shadow:0 1px 3px rgba(0,0,0,.25);");
  b.innerHTML = svg(d, size * 0.6);
  b.title = title;
  b.onmouseenter = () => (b.style.opacity = "1");
  b.onmouseleave = () => (b.style.opacity = ".92");
  return b;
};
// Narrow outlined pill; its label is the NEXT action a click performs.
const pill = (text, css = "") =>
  el("button", "font:10px sans-serif;line-height:15px;padding:0 6px;border:1px solid #bbb;border-radius:8px;background:#fff;color:#333;cursor:pointer;white-space:nowrap;" + css, text);
const light = (c) => /^#/.test(c) && (0.299 * parseInt(c.slice(1, 3), 16) + 0.587 * parseInt(c.slice(3, 5), 16) + 0.114 * parseInt(c.slice(5, 7), 16)) > 200;
const text = (t, top, color) =>
  el("div", `position:absolute;left:4px;top:${top}px;font:10px sans-serif;line-height:12px;color:${color};white-space:nowrap;user-select:none;${light(color) ? "text-shadow:0 0 1.5px #555;" : ""}`, t);
const widest = (col) => Math.max(0, ...[...col.children].map((c) => c.offsetWidth));

// Ticks >= 6 px apart, secondary labels >= 40 px, primary >= 90 px (intervals in seconds).
function ticks(pxPerSec) {
  const step = STEPS.find((s) => s * pxPerSec >= 6) ?? 600;
  const every = (base, px) => base * ([1, 2, 5, 10, 20, 50, 100].find((n) => n * base * pxPerSec >= px) ?? 100);
  const sec = every(step, 40), dec = sec < 0.1 ? 2 : sec < 1 ? 1 : 0;
  return { timeInterval: step, secondaryLabelInterval: sec, primaryLabelInterval: every(sec, 90), formatTimeCallback: (s) => s.toFixed(dec) };
}

function union(segs) {
  const out = [];
  for (const [s, e] of [...segs].sort((a, b) => a[0] - b[0])) {
    if (out.length && s <= out[out.length - 1][1]) out[out.length - 1][1] = Math.max(out[out.length - 1][1], e);
    else out.push([s, e]);
  }
  return out;
}

function render({ model, el: host }) {
  const H = model.get("height"), N = model.get("channels") || 1, D = model.get("duration"), RH = model.get("row_height") || 16;
  const R = N + (N >= 2 ? 1 : 0);  // audio rows drawn: the inputs plus, with >= 2 audios, the mix row (mean of the enabled ones)
  const subs = [], on = (ev, cb) => { model.on(ev, cb); subs.push([ev, cb]); };  // removed on destroy
  const viewId = Symbol("view");  // several views of one model: only the one that started playback makes sound
  const step = model.get("step") || 0.01;
  const expanded = new Set(model.get("expanded") || []);  // annotation names with an open legend column
  let enabled = (model.get("enabled") || []).slice(0, N);
  while (enabled.length < N) enabled.push(true);

  // ---- solo set; `keep` = union of their segments (null = nothing soloed)
  const solos = new Map((model.get("solo") || []).map((k) => (k.startsWith("!") ? [k.slice(1), "inv"] : [k, "keep"])));  // key -> mode
  let keep = null;
  const inKeep = (t) => !keep || keep.some(([a, b]) => t >= a && t <= b);
  const nextStart = (t) => keep && (keep.find(([a]) => a > t) || [null])[0];

  // ---- panel: row 1 time | ◀ play ▶ ; row 2 all/none | fft legend
  const root = el("div", "position:relative;width:100%;background:#fff;");
  const row = (h) => el("div", `position:relative;height:${h}px;display:flex;align-items:center;justify-content:center;gap:10px;`);
  const bar1 = row(BAR_H + 28), bar2 = row(BAR_H);
  const timeOut = el("div", "position:absolute;left:6px;top:18px;font:11px monospace;color:#444;", "0.000 s");
  const back = round(ICON.back, 34, "#1e88e5", `-${step * 1000} ms`), play = round(ICON.play, 48, "#43a047", "play / pause"), fwd = round(ICON.fwd, 34, "#1e88e5", `+${step * 1000} ms`);
  const allBtn = pill("none", "background:#333;color:#fff;border-color:#333;font-weight:600;min-width:44px;letter-spacing:.3px;");
  const fft = pill("FFT", "min-width:44px;font-size:9px;font-weight:700;letter-spacing:1.2px;padding:1px 8px;border-color:#888;color:#222;");
  const left2 = el("div", "position:absolute;left:4px;top:4px;");
  left2.appendChild(allBtn);
  bar1.append(timeOut, back, play, fwd);
  bar2.append(left2, fft);
  const wave = el("div", "position:relative;width:100%;display:flex;align-items:flex-start;");
  const col0 = el("div", "position:relative;flex:none;"), col1 = el("div", "position:relative;flex:none;");
  const wsHost = el("div", "position:relative;flex:1;min-width:0;");
  wave.append(col0, col1, wsHost);
  root.append(bar1, bar2, wave);
  host.appendChild(root);

  const ws = WaveSurfer.create({
    container: wsHost, url: model.get("audio_url"), peaks: model.get("peaks"), duration: D, backend: "WebAudio",
    height: H, splitChannels: R > 1 ? Array.from({ length: R }, () => ({ height: H })) : undefined,
    waveColor: "#9e9e9e", progressColor: "#9e9e9e", cursorColor: "#e00", cursorWidth: 2,
    autoScroll: true, autoCenter: true,
    plugins: [],
  });
  const zoomPlugin = ws.registerPlugin(Zoom.create({ scale: 0.25, maxZoom: 2000, deltaThreshold: 5 }));
  ws.on("scroll", () => (zoomPlugin.oldX = 0));   // upstream caches the pointer time while the mouse is still;
  ws.on("seeking", () => (zoomPlugin.oldX = 0));  // invalidate it whenever the view or position moved
  const wrapper = ws.getWrapper();
  const spec = el("div", `display:none;position:relative;height:${H * R}px;`);
  const overlay = el("div", `position:absolute;left:0;top:0;width:100%;height:${R * H}px;z-index:4;pointer-events:none;`); // functions + white-out
  const timeHost = el("div", `position:relative;height:${TIMELINE_H}px;`);
  const lanes = el("div", "position:relative;width:100%;font-family:sans-serif;user-select:none;");
  wrapper.append(spec, overlay, timeHost, lanes);

  const tl = Timeline.create({ container: timeHost, height: TIMELINE_H, insertPosition: "beforeend", style: { fontSize: "10px", color: "#666" }, ...ticks(wave.clientWidth / D || 30) });
  ws.on("redraw", () => Object.assign(tl.options, ticks(wrapper.scrollWidth / D)));
  ws.registerPlugin(tl);

  // ---- per-audio gains: player -> splitter -> gain_i -> output. Mix = sum(enabled_i * w_i * audio_i).
  const player = ws.getMediaElement(), ctx = player.audioContext, split = ctx.createChannelSplitter(N);
  const maskGain = ctx.createGain();  // solo: silence outside the kept segments by scheduled automation, no buffer copies
  player.getGainNode().disconnect();
  player.getGainNode().connect(maskGain).connect(split);
  const gains = Array.from({ length: N }, (_, i) => { const g = ctx.createGain(); split.connect(g, i); g.connect(ctx.destination); return g; });
  const toggles = [];
  const applyEnabled = () => {
    allBtn.textContent = enabled.every(Boolean) ? "none" : "all";
    const g1 = 1 / Math.max(1, enabled.filter(Boolean).length);  // 1/m over the m selected audios, weights or not
    enabled.forEach((on, i) => {
      gains[i].gain.setTargetAtTime(on ? g1 : 0, ctx.currentTime, 0.01);
      const t = toggles[i];
      if (!t) return;
      t.innerHTML = (on
        ? `<svg width="11" height="11" viewBox="0 0 12 12" style="vertical-align:-1px;margin-right:5px"><rect width="12" height="12" rx="2.5" fill="#43a047"/><path d="M2.5 6.2l2.4 2.4 4.6-4.8" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`
        : `<svg width="11" height="11" viewBox="0 0 12 12" style="vertical-align:-1px;margin-right:5px"><rect x=".75" y=".75" width="10.5" height="10.5" rx="2.5" fill="#fff" stroke="#9e9e9e" stroke-width="1.5"/></svg>`) +
        (model.get("audio_names")?.[i] || "");
      const m = overlay.querySelector(`.mute[data-row="${i}"]`);  // muted rows look like masked gaps
      if (!on && !m) { const d = el("div", `position:absolute;left:0;width:100%;top:${i * H}px;height:${H}px;background:rgba(255,255,255,.85);`); d.className = "mute"; d.dataset.row = i; overlay.appendChild(d); }
      if (on && m) m.remove();
      t.style.color = on ? "#222" : "#999";
      t.style.borderColor = on ? "#9ccc9e" : "#ddd";
      t.title = on ? "playing, click to mute" : "muted, click to play";
    });
  };
  const setEnabled = (arr) => { enabled = arr; model.set("enabled", arr.slice()); model.save_changes(); applyEnabled(); renderMix(); };
  allBtn.onclick = () => { const on = !enabled.every(Boolean); setEnabled(enabled.map(() => on)); };
  on("change:weighted", applyEnabled);
  on("change:enabled", () => { enabled = (model.get("enabled") || []).slice(0, N); while (enabled.length < N) enabled.push(true); applyEnabled(); renderMix(); });
  on("change:spectrograms", () => draw());

  // ---- solo: silence outside the kept segments (masked buffer) and skip the gaps while playing
  let orig = null;
  // Rendering buffer = the inputs plus the mix row (mean of the enabled ones); playback keeps the N-channel buffer.
  const withMix = (buf) => {
    if (R === N) return buf;
    const out = ctx.createBuffer(R, buf.length, buf.sampleRate), mix = out.getChannelData(N);
    const g1 = 1 / Math.max(1, enabled.filter(Boolean).length);
    for (let c = 0; c < N; c++) {
      const src = buf.getChannelData(c);
      out.copyToChannel(src, c);
      if (enabled[c]) for (let i = 0; i < src.length; i++) mix[i] += src[i] * g1;
    }
    return out;
  };
  let mixTimer;
  const renderMix = () => { if (!orig || R === N) return; clearTimeout(mixTimer); mixTimer = setTimeout(() => { ws.decodedData = withMix(orig); ws.renderer.render(ws.decodedData); }, 30); };
  const segsOf = (key) => {
    const [kind, gi, lab] = key.split(":"), g = (model.get("groups") || [])[+gi] || { segments: [] };
    return union(g.segments.filter((s) => kind === "g" || s.label === lab).map((s) => [s.start, s.end]));
  };
  const complement = (segs) => {  // gaps of a sorted, merged list within [0, D]
    const out = [];
    let prev = 0;
    for (const [s, e] of segs) { if (s > prev) out.push([prev, s]); prev = Math.max(prev, e); }
    if (D > prev) out.push([prev, D]);
    return out;
  };
  const intersect = (a, b) => {
    const out = [];
    for (const [s1, e1] of a) for (const [s2, e2] of b) { const s = Math.max(s1, s2), e = Math.min(e1, e2); if (e > s) out.push([s, e]); }
    return union(out);
  };
  const maskOf = (key, mode) => (mode === "inv" ? complement(segsOf(key)) : segsOf(key));
  const soloState = () => [...solos].map(([k, m]) => (m === "inv" ? "!" + k : k));
  // Schedule the mask gain from the current position: 1 inside kept segments, 0 outside, 5 ms ramps.
  const scheduleMask = () => {
    const g = maskGain.gain, now = ctx.currentTime, t0 = ws.getCurrentTime();
    g.cancelScheduledValues(now);
    if (!keep) { g.setValueAtTime(1, now); return; }
    g.setValueAtTime(inKeep(t0) ? 1 : 0, now);
    for (const [s, e] of keep) {
      if (s > t0) g.setTargetAtTime(1, now + (s - t0), FADE / 3);
      if (e > t0) g.setTargetAtTime(0, now + (e - t0), FADE / 3);
    }
  };
  const applySolo = () => {
    keep = null;
    for (const [k, mode] of solos) keep = keep ? intersect(keep, maskOf(k, mode)) : maskOf(k, mode);
    for (const d of [...col0.children, ...col1.children]) if (d.dataset.key) {
      const mode = solos.get(d.dataset.key);
      d.style.outline = mode === "keep" ? "1px solid #e00" : mode === "inv" ? "1px dashed #e00" : "";
    }
    drawMask();
    scheduleMask();
  };
  const drawMask = () => {  // white-out the gaps on every audio row (wave and spectrogram alike)
    overlay.querySelectorAll(".gap").forEach((d) => d.remove());
    if (!keep) return;
    let prev = 0;
    for (const [s, e] of [...keep, [D, D]]) {
      if (s > prev) overlay.appendChild(Object.assign(el("div", `position:absolute;top:0;height:100%;left:${(100 * prev) / D}%;width:${(100 * (s - prev)) / D}%;background:rgba(255,255,255,.92);`), { className: "gap" }));
      prev = Math.max(prev, e);
    }
  };
  let pendingPlay = false;
  player.addEventListener("canplay", () => {
    if (player.buffer && !orig) { orig = player.buffer; ws.decodedData = withMix(orig); ws.renderer.render(ws.decodedData); }
    if (pendingPlay) { pendingPlay = false; ws.play(); }
  });
  const safePlay = () => { model._wavlensActive = viewId; ctx.resume?.(); if (orig) ws.play(); else pendingPlay = true; };

  // ---- dim
  const dimmed = new Set(model.get("dimmed") || []);
  const segDivs = [];
  const applyDim = () => {
    for (const d of [...col0.children, ...col1.children]) if (d.dataset.key) d.style.opacity = dimmed.has(d.dataset.key) ? ".25" : "";
    for (const d of segDivs) d.style.opacity = dimmed.has(`g:${d.dataset.group}`) || dimmed.has(`l:${d.dataset.group}:${d.dataset.label}`) ? ".25" : "";
  };
  const clickable = (d, key) => {  // click cycles: neutral -> keep -> complement -> neutral
    d.dataset.key = key;
    d.style.cursor = "pointer";
    d.title = "click: keep its segments / keep the rest / all";
    d.onclick = () => {
      if (d.dataset.dragged) { delete d.dataset.dragged; return; }
      const mode = solos.get(key);
      if (!mode) solos.set(key, "keep"); else if (mode === "keep") solos.set(key, "inv"); else solos.delete(key);
      model.set("solo", soloState()); model.save_changes(); applySolo();
    };
  };

  // ---- drawing (original time axis)
  const lines = (fns, top, height) => {
    const s = document.createElementNS(NS, "svg");
    s.setAttribute("viewBox", "0 0 1 1");
    s.setAttribute("preserveAspectRatio", "none");
    s.setAttribute("class", "fn");
    s.style.cssText = `position:absolute;left:0;top:${top}px;width:100%;height:${height}px;pointer-events:none;overflow:visible;`;
    for (const f of fns) {
      const pts = f.x.map((x, i) => `${x},${1 - f.y[i]}`).join(" ");
      for (const [stroke, w, op] of [["#fff", 3, ".6"], [f.color, 1, ".9"]]) {  // halo keeps it readable on the spectrogram
        const p = document.createElementNS(NS, "polyline");
        p.setAttribute("points", pts);
        p.setAttribute("style", `fill:none;stroke:${stroke};stroke-width:${w};vector-effect:non-scaling-stroke;opacity:${op}`);
        p.innerHTML = `<title>${f.name}</title>`;
        s.appendChild(p);
      }
    }
    return s;
  };
  const draw = () => {
    col0.innerHTML = "";
    col1.innerHTML = "";
    lanes.innerHTML = "";
    overlay.querySelectorAll(".fn").forEach((d) => d.remove());
    spec.innerHTML = "";
    toggles.length = 0;
    segDivs.length = 0;
    const fns = model.get("functions") || [], groups = model.get("groups") || [];
    const pct = (t) => `${(100 * t) / D}%`;
    (model.get("spectrograms") || []).forEach((src, i) => {
      if (src) spec.appendChild(Object.assign(el("img", `position:absolute;left:0;top:${i * H}px;width:100%;height:${H}px;`), { src }));
    });
    for (let i = 0; i < R; i++) {
      if (i < N) {
        const t = pill("", `position:absolute;left:4px;top:${i * H + 3}px;padding:0 7px 0 6px;border-radius:9px;line-height:16px;`);
        t.onclick = () => setEnabled(enabled.map((on, k) => (k === i ? !on : on)));
        toggles.push(t);
        col0.appendChild(t);
      } else {  // mix row: not selectable, framed in blue
        col0.appendChild(el("div", `position:absolute;left:4px;top:${i * H + 3}px;font:600 10px sans-serif;color:#1e88e5;padding:1px 7px;border:1px solid #1e88e5;border-radius:9px;line-height:14px;`, "mix"));
        overlay.appendChild(el("div", `position:absolute;left:0;width:100%;top:${i * H}px;height:${H}px;box-sizing:border-box;border-top:1.5px solid #1e88e5;border-bottom:1.5px solid #1e88e5;background:rgba(30,136,229,.04);`));
      }
      const own = fns.filter((f) => f.row === i || f.row === -1 || f.row == null);
      if (own.length) overlay.appendChild(lines(own, i * H, H));
    }
    applyEnabled();
    const nrows = groups.reduce((n, g) => n + g.rows.length, 0);
    lanes.style.height = `${nrows * RH}px`;
    const laneTop = lanes.offsetTop;  // measured: wherever the timeline really ends
    let top = 0;
    groups.forEach((g, gi) => {
      const shared = fns.filter((f) => f.row === -1 || f.row == null);
      if (shared.length) lanes.appendChild(lines(shared, top, g.rows.length * RH));
      const color = {};
      for (const s of g.segments) {
        color[s.label] = s.color;
        const h = Math.max(RH - 2, 2);
        const d = el("div",
          `position:absolute;top:${top + g.rows.indexOf(s.label) * RH + 1}px;height:${h}px;left:${pct(s.start)};width:${pct(s.end - s.start)};` +
          `background:${s.color};color:#fff;border-radius:2px;overflow:hidden;white-space:nowrap;text-indent:3px;box-sizing:border-box;line-height:${h}px;font-size:${Math.min(11, RH - 4)}px;`,
          s.label);
        d.title = `${s.label} ${s.start.toFixed(2)}–${s.end.toFixed(2)}s`;
        d.dataset.group = gi;
        d.dataset.label = s.label;
        segDivs.push(d);
        lanes.appendChild(d);
      }
      const y = (i) => laneTop + top + i * RH + Math.max(0, (RH - 12) / 2);
      const open = expanded.has(g.name);
      const tog = el("button", `position:absolute;left:2px;top:${y(0) - 1}px;width:13px;height:13px;padding:0;border:0;background:none;color:#777;cursor:pointer;font:11px/13px sans-serif;`, open ? "▾" : "▸");
      tog.title = open ? "hide this annotation's speakers" : "show this annotation's speakers";
      tog.onclick = () => { open ? expanded.delete(g.name) : expanded.add(g.name); model.set("expanded", [...expanded]); model.save_changes(); draw(); };
      col0.appendChild(tog);
      const nameEl = text(g.name, y(0), "#555");
      nameEl.style.left = "16px";
      clickable(nameEl, `g:${gi}`);
      draggable(nameEl, gi);
      col0.appendChild(nameEl);
      if (open) g.rows.forEach((r, i) => { if (r) clickable(col1.appendChild(text(r, y(i), color[r])), `l:${gi}:${r}`); });
      top += g.rows.length * RH;
    });
    // columns: as wide as their widest entry (column 0 also fits the all/none button); column 1 collapses when nothing is expanded
    const h = laneTop + top;
    col0.style.height = col1.style.height = `${h}px`;
    col0.style.width = `${Math.max(...[...col0.children].map((c) => c.offsetLeft + c.offsetWidth), allBtn.offsetWidth) + 8}px`;
    col1.style.width = `${widest(col1) + 8}px`;
    col1.style.display = col1.children.length ? "" : "none";
    applySolo();
    applyDim();
  };
  // Drag an annotation name up/down to reorder the lane groups; a click (< 5 px) is left to the mask cycle.
  const marker = el("div", "position:absolute;left:0;right:0;height:0;border-top:2px solid #1e88e5;display:none;z-index:9;pointer-events:none;");
  wave.appendChild(marker);
  const draggable = (d, gi) => {
    d.style.touchAction = "none";
    d.onpointerdown = (e0) => {
      const groups = model.get("groups") || [];
      let y0 = e0.clientY, moved = false, target = gi;
      const bounds = () => {  // y of each group boundary (top of group k), in wave coordinates
        const laneTop = lanes.offsetTop, out = [laneTop];
        for (const g of groups) out.push(out[out.length - 1] + g.rows.length * RH);
        return out;
      };
      const move = (e) => {
        if (!moved && Math.abs(e.clientY - y0) < 5) return;
        moved = true;
        d.style.opacity = ".4";
        const b = bounds(), y = e.clientY - wave.getBoundingClientRect().top;
        target = 0;
        for (let k = 0; k < groups.length; k++) if (y > (b[k] + b[k + 1]) / 2) target = k + 1;
        marker.style.top = `${b[Math.min(target, groups.length)]}px`;
        marker.style.display = "";
      };
      const up = () => {
        document.removeEventListener("pointermove", move);
        document.removeEventListener("pointerup", up);
        marker.style.display = "none";
        d.style.opacity = "";
        if (!moved) return;
        d.dataset.dragged = "1";  // swallow the click that follows a drag
        const dest = target > gi ? target - 1 : target;
        if (dest === gi) return;
        const order = groups.map((_, k) => k);
        order.splice(dest, 0, order.splice(gi, 1)[0]);
        const remap = (key) => { const [kind, i, ...rest] = key.split(":"); return [kind, order.indexOf(+i), ...rest].join(":"); };
        for (const [k, m] of [...solos]) { solos.delete(k); solos.set(remap(k), m); }
        for (const k of [...dimmed]) { dimmed.delete(k); dimmed.add(remap(k)); }
        model.set("groups", order.map((k) => groups[k]));
        model.set("solo", soloState());
        model.set("dimmed", [...dimmed]);
        model.save_changes();
        draw();
        applySolo();
      };
      document.addEventListener("pointermove", move);
      document.addEventListener("pointerup", up);
    };
  };
  draw();
  ws.on("ready", draw);
  ["groups", "functions", "audio_names"].forEach((k) => on(`change:${k}`, () => {
    const groups = model.get("groups") || [];
    for (const key of [...solos.keys()]) {  // drop solos whose annotation / speaker no longer exists
      const [kind, gi, lab] = key.split(":"), g = groups[+gi];
      if (!g || (kind === "l" && !g.rows.includes(lab))) solos.delete(key);
    }
    for (const key of [...dimmed]) { const [kind, gi, lab] = key.split(":"), g = groups[+gi]; if (!g || (kind === "l" && !g.rows.includes(lab))) dimmed.delete(key); }
    draw();
    applySolo();
  }));
  // Rendered while hidden (collapsed output): the gutter measures 0; re-fit when it becomes visible.
  let seenWidth = root.clientWidth;
  const ro = new ResizeObserver(() => { if (seenWidth === 0 && root.clientWidth > 0) draw(); seenWidth = root.clientWidth; });
  ro.observe(root);
  if (!(model.get("spectrograms") || []).some(Boolean)) { fft.disabled = true; fft.title = "no spectrogram (matplotlib missing)"; fft.style.opacity = ".4"; }
  on("change:expanded", () => { expanded.clear(); for (const n of model.get("expanded") || []) expanded.add(n); draw(); });
  on("change:solo", () => { solos.clear(); for (const k of model.get("solo") || []) solos.set(k.startsWith("!") ? k.slice(1) : k, k.startsWith("!") ? "inv" : "keep"); applySolo(); });
  on("change:dimmed", () => { dimmed.clear(); for (const k of model.get("dimmed") || []) dimmed.add(k); applyDim(); });

  // ---- controls
  let showSpec = model.get("view") === "fft";
  const applyView = () => {
    fft.textContent = showSpec ? "WAVE" : "FFT";
    for (const c of wrapper.children) if (/canvases|progress/.test(c.getAttribute("part") || "")) c.style.display = showSpec ? "none" : "";
    spec.style.display = showSpec ? "" : "none";
  };
  fft.onclick = () => { showSpec = !showSpec; applyView(); };
  const nudge = (d) => ws.setTime(Math.min(D, Math.max(0, ws.getCurrentTime() + d)));
  const playPause = () => (ws.isPlaying() ? ws.pause() : safePlay());
  back.onclick = () => nudge(-step);
  fwd.onclick = () => nudge(step);
  play.onclick = playPause;
  root.tabIndex = 0;
  root.style.outline = "none";
  root.addEventListener("pointerdown", () => root.focus());
  root.addEventListener("keydown", (e) => {
    const act = { Space: playPause, ArrowLeft: () => nudge(-step), ArrowRight: () => nudge(step) }[e.code];
    if (act) { e.preventDefault(); e.stopPropagation(); act(); }
  });
  let lastClick = 0;
  ws.on("interaction", () => { const now = performance.now(); if (now - lastClick > 300) safePlay(); lastClick = now; }); // click = seek + play; a double-click does not restart
  let hold;  // while wheel-zooming, keep the view under the pointer even during playback
  wsHost.addEventListener("wheel", () => {
    ws.options.autoScroll = false;
    clearTimeout(hold);
    hold = setTimeout(() => (ws.options.autoScroll = true), 400);
  }, { passive: true });
  ws.once("ready", () => {  // initial state from arguments
    applyView();
    applySolo();
    const z = model.get("zoom");
    if (z > 0) ws.zoom(z);
    const t0 = model.get("current_time");
    if (t0 > 0) ws.setTime(Math.min(D, t0));
    if (model.get("autoplay")) safePlay();
  });

  // ---- sync with Python; while soloed, playback skips the gaps
  let last = 0, flush;
  let sending = false;
  const sendTime = (t) => { sending = true; model.set("current_time", Math.min(D, Math.max(0, t))); model.save_changes(); sending = false; };
  ws.on("timeupdate", (t) => {
    if (keep && ws.isPlaying() && !inKeep(t)) {
      const n = nextStart(t);
      if (n == null) { ws.pause(); ws.setTime(keep[keep.length - 1][1]); return; }
      ws.setTime(n);
      return;
    }
    timeOut.textContent = `${t.toFixed(3)} s`;
    clearTimeout(flush);
    if (performance.now() - last < 100) { flush = setTimeout(() => sendTime(ws.getCurrentTime()), 120); return; }  // trailing flush: the last nudge always lands
    last = performance.now();
    sendTime(t);
  });
  ws.on("play", () => {
    if (keep && !inKeep(ws.getCurrentTime())) ws.setTime(nextStart(ws.getCurrentTime()) ?? keep[0][0]);
    play.innerHTML = svg(ICON.pause, 29); model.set("playing", true); model.save_changes();
  });
  ws.on("pause", () => {
    const t = Math.min(D, ws.getCurrentTime());
    ws.setTime(t);  // cursor can be a frame behind the clock: redraw it at the exact position
    timeOut.textContent = `${t.toFixed(3)} s`;
    play.innerHTML = svg(ICON.play, 29);
    clearTimeout(flush);
    model.set("playing", false); sendTime(t);
  });
  on("change:current_time", () => {  // from Python only (our own sets are flagged)
    const t = model.get("current_time");
    if (sending || !Number.isFinite(t)) return;
    ws.setTime(Math.min(D, Math.max(0, t)));
  });
  on("change:playing", () => {  // another view (or Python) started/stopped playback
    const p = model.get("playing");
    if (p && model._wavlensActive && model._wavlensActive !== viewId) return;  // that view makes the sound; we only follow current_time
    if (p !== ws.isPlaying()) p ? safePlay() : ws.pause();
  });
  on("change:current_time", () => { if (!sending && model.get("playing") && model._wavlensActive !== viewId && !ws.isPlaying()) ws.setTime(model.get("current_time")); });

  return () => {  // destroy: stop sound, release the AudioContext, drop every listener and timer
    clearTimeout(hold); clearTimeout(flush);
    ro.disconnect();
    try { ws.pause(); } catch (e) { /* already gone */ }
    for (const [ev, cb] of subs) model.off(ev, cb);
    if (model._wavlensActive === viewId) model._wavlensActive = null;
    ws.destroy();
    player.destroy?.();
    if (ctx.state !== "closed") ctx.close().catch(() => {});
  };
}

export default { render };
