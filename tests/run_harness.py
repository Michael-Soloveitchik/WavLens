"""Playwright harness for wavlens.js: serve repo root, render, seek, zoom, fft, ticks, legend, transport.

Run: nice python3 tests/run_harness.py   (wavlens/wavlens.js is an offline esbuild bundle)
"""

from __future__ import annotations

import functools
import http.server
import json
import re
import socketserver
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "out"

# Walk every shadow root to find nodes matching a selector.
JS_DEEP_QUERY = """
(sel) => {
  const out = [];
  const walk = (root) => {
    for (const n of root.querySelectorAll(sel)) out.push(n);
    for (const n of root.querySelectorAll('*')) if (n.shadowRoot) walk(n.shadowRoot);
  };
  walk(document);
  return out;
}
"""

# Rows inside wavesurfer's wrapper, in order: canvases/progress/cursor (part=...),
# then wavlens's own spec host, timeline host, lanes host (last own div).
JS_STATE = """
() => {
  const deep = %s;
  const wrapper = deep('.wrapper')[0] || deep('[part="wrapper"]')[0];
  const scroll = deep('.scroll')[0] || deep('[part="scroll"]')[0];
  const kids = wrapper ? [...wrapper.children] : [];
  const own = kids.filter((c) => !c.getAttribute('part') && c.tagName === 'DIV');  // skip function-plot svgs
  // Own divs in order: spectrogram host, overlay (functions + join marks; newer builds), timeline host, lanes host.
  const laneHost = own[own.length - 1] || null;
  const timeHost = own[own.length - 2] || null;
  const specHost = own[0] || null;
  const maskHost = own.length >= 4 ? own[1] : null;
  const overlay = maskHost;
  const segDivs = laneHost ? [...laneHost.querySelectorAll('div[title]')] : [];
  const waveParts = kids.filter((c) => /canvases|progress/.test(c.getAttribute('part') || ''));
  const canvases = deep('[part="canvases"] canvas');
  // Spectrogram rows are Python-rendered <img> data URLs stretched to the wrapper (older builds: a plugin <canvas>).
  const specCanvas = specHost ? specHost.querySelector('canvas, img') : null;
  const specReady = specHost ? [...specHost.querySelectorAll('canvas, img')].filter((e) => e.tagName === 'CANVAS' || (e.complete && e.naturalWidth > 0)).length : 0;
  const sw = (e) => (e ? e.scrollWidth : null);
  // Left gutter (light DOM, never scrolls): wave row = root's 3rd child; column 0 = ☑ toggles + annotation
  // group names (gray divs), column 1 = per-row colored legend labels (side/both only), then the wavesurfer host.
  const waveRow = document.querySelector('#widget > div > div:nth-child(3)');
  const col0 = waveRow ? waveRow.children[0] : null, col1 = waveRow ? waveRow.children[1] : null;
  const col0Divs = col0 ? [...col0.querySelectorAll('div')] : [];
  const col1Divs = col1 ? [...col1.querySelectorAll('div')] : [];
  const gutterKids = [...(col0 ? col0.children : []), ...(col1 ? col1.children : [])];
  // Timeline plugin puts label text directly inside notch divs (part="timeline-notch timeline-notch-primary|secondary").
  const tlLabels = timeHost
    ? [...timeHost.querySelectorAll('[part*="notch"]')].map((n) => n.textContent.trim()).filter(Boolean)
    : [];
  const tlNotches = timeHost ? timeHost.querySelectorAll('[part*="notch"]').length : 0;
  const bar = document.querySelector('#widget > div > div');
  const readout = bar ? [...bar.children].find((c) => /monospace/.test(c.style.font)) : null;
  const allButtons = [...document.querySelectorAll('#widget button')];
  const buttons = allButtons.map((b) => b.textContent);
  // Per-audio on/off toggles: pills in gutter column 0 (SVG checkbox + name; older builds: ☑/☐ text).
  const col0Early = document.querySelector('#widget > div > div:nth-child(3) > div');
  const toggles = (col0Early ? [...col0Early.querySelectorAll('button')].filter((b) => b.querySelector('svg')) : allButtons.filter((b) => /^[☑☐]/.test(b.textContent)))
    .map((b) => ({ text: b.textContent.trim(), opacity: b.style.opacity, color: b.style.color, title: b.title,
                   checked: !!b.querySelector('svg path'), x: b.getBoundingClientRect().x, y: b.getBoundingClientRect().y }));
  // Per-group ▸/▾ expanders (column 0) open that annotation's speaker labels in column 1.
  const expanders = col0Early ? [...col0Early.querySelectorAll('button')].filter((b) => /[▸▾]/.test(b.textContent))
    .map((b) => ({ text: b.textContent, title: b.title, x: b.getBoundingClientRect().x, y: b.getBoundingClientRect().y })) : [];
  const Dur = window.wavlensModel ? window.wavlensModel.get('duration') : 1;
  const secs = (d) => { const l = parseFloat(d.style.left) / 100 * Dur, w = parseFloat(d.style.width) / 100 * Dur; return [Math.round(l * 1000) / 1000, Math.round((l + w) * 1000) / 1000]; };
  const modeBtn = allButtons.find((b) => /^(all|none)$/.test(b.textContent));
  const legendBtn = allButtons.find((b) => /^(side|both|inline)$/.test(b.textContent));
  const canvasesPart = deep('[part="canvases"]')[0];
  const channelHeights = canvasesPart ? [...canvasesPart.children].map((c) => c.getBoundingClientRect().height) : [];
  return {
    ready: window.wavlensReady, error: window.wavlensError,
    now_ms: performance.now(),
    wrapper_width: wrapper ? wrapper.getBoundingClientRect().width : null,
    wrapper_height: wrapper ? wrapper.getBoundingClientRect().height : null,
    wrapper_scroll_width: sw(wrapper), timeline_scroll_width: sw(timeHost), lanes_scroll_width: sw(laneHost),
    scroll_left: scroll ? scroll.scrollLeft : null,
    waveform_canvases: canvases.length,
    waveform_canvas_widths: canvases.map((c) => c.width),
    marked_canvases: canvases.filter((c) => c.dataset.wavlensMark).length,
    waveform_hidden: waveParts.length > 0 && waveParts.every((c) => c.style.display === 'none'),
    spec_visible: !!specHost && specHost.style.display !== 'none' && specHost.offsetHeight > 0,
    spec_canvases: specReady,
    spec_kind: specCanvas ? specCanvas.tagName.toLowerCase() : null,
    spec_canvas_width: specCanvas ? specCanvas.getBoundingClientRect().width : null,
    spec_client_width: specCanvas ? specCanvas.clientWidth : null,
    spec_img_visible: !!specCanvas && specCanvas.getClientRects().length > 0,
    lane_segments: segDivs.length,
    lane_segments_with_text: segDivs.filter((d) => d.textContent.trim()).length,
    lane_groups: window.wavlensModel ? (window.wavlensModel.get('groups') || []).length : null,
    fn_polylines_audio: (overlay || wrapper) ? (overlay || wrapper).querySelectorAll(':scope > svg.fn polyline, :scope > svg.wavlens-fn polyline').length : 0,
    gap_whiteouts: overlay ? overlay.querySelectorAll('.gap').length : null,
    gaps_s: overlay ? [...overlay.querySelectorAll('.gap')].map(secs) : [],
    mute_rows: overlay ? [...overlay.querySelectorAll('.mute')].map((d) => d.dataset.row) : [],
    fn_polylines_lanes: laneHost ? laneHost.querySelectorAll('svg polyline').length : 0,
    fn_per_audio_row: (overlay || wrapper) ? [...(overlay || wrapper).querySelectorAll(':scope > svg.fn')]
      .sort((a, b) => parseFloat(a.style.top) - parseFloat(b.style.top))
      .map((s) => [...s.querySelectorAll('polyline')].map((p) => p.querySelector('title')?.textContent)) : [],
    fn_per_lane_group: laneHost ? [...laneHost.querySelectorAll(':scope > svg')]
      .map((s) => [...s.querySelectorAll('polyline')].map((p) => p.querySelector('title')?.textContent)) : [],
    row_labels: col0Divs.map((d) => d.textContent),
    colored_labels: col1Divs.filter((d) => d.style.color).map((d) => d.textContent),
    colored_label_colors: col1Divs.map((d) => d.style.color),
    gutter_widths: [col0 ? col0.offsetWidth : null, col1 ? col1.offsetWidth : null],
    col1_visibility: col1 ? getComputedStyle(col1).visibility : null,
    gutter_children: gutterKids.length,
    gutter_xs: gutterKids.map((d) => Math.round(d.getBoundingClientRect().x)),
    timeline_labels: tlLabels, timeline_notches: tlNotches,
    readout: readout ? readout.textContent : null,
    buttons, toggles,
    mode_button: modeBtn ? modeBtn.textContent : null,
    legend_button: legendBtn ? legendBtn.textContent : null,
    play_button: allButtons[1] ? (allButtons[1].textContent || allButtons[1].querySelector('path')?.getAttribute('d')?.slice(0, 24)) : null,
    transport_titles: allButtons.slice(0, 3).map((b) => b.title),
    transport_have_svg_path: allButtons.slice(0, 3).every((b) => !!b.querySelector('svg path')),
    mask_children: maskHost ? maskHost.children.length : null,
    join_marks: overlay ? [...overlay.children].filter((d) => d.tagName === 'DIV' && /dashed/.test(d.style.borderLeft)).length : null,
    spec_tiles: specHost ? specHost.children.length : 0,
    spec_tile_imgs: specHost ? specHost.querySelectorAll('img').length : 0,
    timeline_bottom: timeHost ? timeHost.getBoundingClientRect().bottom : null,
    lanes_top: laneHost ? laneHost.getBoundingClientRect().top : null,
    last_ruler_label: tlLabels.length ? tlLabels[tlLabels.length - 1] : null,
    buffers: window.wavlensBuffers || [],
    name_labels: [...(col0 ? col0.children : []), ...(col1 ? col1.children : [])].filter((d) => d.dataset.key)
      .map((d) => { const r = d.getBoundingClientRect(); return { key: d.dataset.key, text: d.textContent, opacity: d.style.opacity, outline: d.style.outline,
        visible: r.height > 0 && getComputedStyle(d).visibility !== 'hidden', top: r.top, center: r.top + r.height / 2 }; }),
    seg_styles: segDivs.map((d) => ({ group: d.dataset.group, label: d.dataset.label, opacity: d.style.opacity, color: d.style.color })),
    lane_segments_transparent: segDivs.filter((d) => d.style.color === 'transparent').length,
    enabled: window.wavlensModel && window.wavlensModel.get('enabled'),
    lanes_height: laneHost ? laneHost.offsetHeight : null,
    lane_rows: window.wavlensModel ? (window.wavlensModel.get('groups') || []).reduce((n, g) => n + g.rows.length, 0) : null,
    canvases_height: canvasesPart ? canvasesPart.getBoundingClientRect().height : null,
    channel_heights: channelHeights,
    label_xs: col0Divs.map((d) => Math.round(d.getBoundingClientRect().x)),
    scroll_y: window.scrollY,
    doc_keys: window.docKeys ?? null,
    current_time: window.wavlensModel && window.wavlensModel.get('current_time'),
    playing: window.wavlensModel && window.wavlensModel.get('playing'),
    legend: window.wavlensModel && window.wavlensModel.get('legend'),
    expanded: window.wavlensModel && window.wavlensModel.get('expanded'),
    expanders,
    group_names: window.wavlensModel ? (window.wavlensModel.get('groups') || []).map((g) => g.name) : [],
    col1_display: col1 ? getComputedStyle(col1).display : null,
    col1_labels: col1Divs.filter((d) => d.dataset.key).map((d) => ({ key: d.dataset.key, text: d.textContent })),
    name_divs: col0Divs.filter((d) => d.dataset.key).map((d) => { const r = d.getBoundingClientRect();
      return { key: d.dataset.key, text: d.textContent, x: r.x, y: r.y, h: r.height, outline: d.style.outline, left: d.style.left }; }),
    seg_top_by_group: segDivs.reduce((acc, d) => { const t = parseFloat(d.style.top); acc[d.dataset.group] = Math.min(acc[d.dataset.group] ?? 1e9, t); return acc; }, {}),
    solo: window.wavlensModel && window.wavlensModel.get('solo'),
    log_tail: (window.wavlensLog || []).slice(-5),
  };
}
""" % JS_DEEP_QUERY

JS_MARK_CANVASES = f"() => ({JS_DEEP_QUERY})('[part=\"canvases\"] canvas').forEach((c) => (c.dataset.wavlensMark = '1'))"

JS_READOUT = "[...document.querySelectorAll('#widget div')].find((d) => /monospace/.test(d.style.font))"

# In-page sampler: [elapsed_ms, readout seconds, model.current_time] every `every` ms for `total` ms.
JS_SAMPLE = """
async ([total, every]) => {
  const ro = %s;
  const out = [], t0 = performance.now();
  while (performance.now() - t0 < total) {
    out.push([Math.round(performance.now() - t0), parseFloat(ro.textContent), window.wavlensModel.get('current_time')]);
    await new Promise((r) => setTimeout(r, every));
  }
  return out;
}
""" % JS_READOUT

# In-page: Space while playing; readout right after the keydown, on the next frame, and 100 ms later.
JS_SPACE_PAUSE = """
() => new Promise((res) => {
  const ro = %s, root = document.querySelector('#widget > div');
  const before = parseFloat(ro.textContent);
  root.dispatchEvent(new KeyboardEvent('keydown', { code: 'Space', key: ' ', bubbles: true, cancelable: true }));
  const sync = parseFloat(ro.textContent);
  requestAnimationFrame(() => {
    const next_frame = parseFloat(ro.textContent);
    setTimeout(() => res({ before, sync, next_frame, after_100ms: parseFloat(ro.textContent),
                           playing_after_100ms: window.wavlensModel.get('playing') }), 100);
  });
})
""" % JS_READOUT

# In-page: ◀ click while playing; readout just before, synchronously after, and 300 ms later.
JS_BACK_WHILE_PLAYING = """
() => new Promise((res) => {
  const ro = %s, [back, , fwd] = document.querySelectorAll('#widget button');
  const r = () => parseFloat(ro.textContent);
  const before = r();           // last rAF readout: up to one frame behind the live clock
  back.click();
  const after = r();            // live - step
  // Calibrated: a net-zero pair (▶ ◀) syncs the readout to the live clock, then one ◀ in the same tick.
  fwd.click(); back.click();
  const synced = r();
  back.click();
  const stepped = r();
  setTimeout(() => res({ before, after, delta_raw: Math.round(1000 * (after - before)) / 1000,
                         synced, stepped, delta_calibrated: Math.round(1000 * (stepped - synced)) / 1000,
                         later_300ms: r(), playing: window.wavlensModel.get('playing') }), 300);
})
""" % JS_READOUT

# In-page: click the waveform at concatenated time u (fraction of the wrapper width = u / effective duration,
# inferred from the ruler's px/s), return the readout right after the click (original seconds).
JS_SEEK_CLICK = """
(u) => {
  const deep = %s;
  const ro = %s;
  const wrapper = deep('.wrapper')[0], canvases = deep('[part="canvases"]')[0];
  const kids = [...wrapper.children].filter((c) => !c.getAttribute('part') && c.tagName === 'DIV');
  const timeHost = kids[kids.length - 2];
  // Effective duration from the ruler: two labelled notches give px per second.
  const notches = [...timeHost.querySelectorAll('[part*="notch"]')].filter((n) => n.textContent.trim());
  const [n0, n1] = [notches[0], notches[notches.length - 1]];
  const pxPerSec = (parseFloat(n1.style.left) - parseFloat(n0.style.left)) / (parseFloat(n1.textContent) - parseFloat(n0.textContent));
  const r = wrapper.getBoundingClientRect(), c = canvases.getBoundingClientRect();
  const x = r.left + u * pxPerSec, y = c.top + c.height / 4;
  const before = ro.textContent;
  wrapper.dispatchEvent(new MouseEvent('click', { clientX: x, clientY: y, bubbles: true, cancelable: true, composed: true }));
  return { u, px_per_sec: Math.round(pxPerSec * 100) / 100, effective_duration_est: Math.round((r.width / pxPerSec) * 100) / 100,
           readout_before: before, readout_after_click: ro.textContent, model_current_time: window.wavlensModel.get('current_time') };
}
""" % (JS_DEEP_QUERY, JS_READOUT)

# In-page: per-view state for multi-view pages (#widget, #widget2, ...) plus AudioContext states.
JS_VIEWS = """
() => {
  const views = [...document.querySelectorAll('[id^="widget"]')].map((host) => {
    const ro = [...host.querySelectorAll('div')].find((d) => /monospace/.test(d.style.font));
    const play = host.querySelectorAll('button')[1];
    const d = play ? play.querySelector('path')?.getAttribute('d') || '' : '';
    return { id: host.id, readout: ro ? ro.textContent : null, play_icon: d.startsWith('M6 5h4') ? 'pause' : d.startsWith('M8 5v14') ? 'play' : d.slice(0, 12) };
  });
  return { views, contexts: (window.wavlensContexts || []).map((c) => c.state),
           playing: window.wavlensModel.get('playing'), current_time: window.wavlensModel.get('current_time'),
           log_len: (window.wavlensLog || []).length };
}
"""

# In-page latency: ms from the action to the first readout text change (-1 if none within 2 s).
JS_LATENCY = """
(action) => new Promise((res) => {
  const ro = %s;
  const root = document.querySelector('#widget > div');
  const mo = new MutationObserver(() => { mo.disconnect(); res(Math.round(100 * (performance.now() - t0)) / 100); });
  mo.observe(ro, { characterData: true, childList: true, subtree: true });
  const t0 = performance.now();
  if (action === 'play') document.querySelectorAll('#widget button')[1].click();
  else root.dispatchEvent(new KeyboardEvent('keydown', { code: action, key: action, bubbles: true, cancelable: true }));
  setTimeout(() => { mo.disconnect(); res(-1); }, 2000);
})
""" % JS_READOUT


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve(root: Path) -> tuple[socketserver.TCPServer, int]:
    handler = functools.partial(QuietHandler, directory=str(root))
    socketserver.TCPServer.allow_reuse_address = True
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def wait_timings(page, timeout_s: float = 60.0) -> dict:
    """Poll until the waveform canvas and the lane segments exist; return ms since navigation start.

    The first canvases seen are tagged so a later re-render (decode swap) is detectable."""
    t = {"canvas_ms": None, "lanes_ms": None}
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        s = page.evaluate(JS_STATE)
        if s["error"]:
            raise RuntimeError(s["error"])
        if t["canvas_ms"] is None and s["waveform_canvases"] > 0:
            t["canvas_ms"] = round(s["now_ms"])
            page.evaluate(JS_MARK_CANVASES)
        if t["lanes_ms"] is None and s["lane_segments"] > 0:
            t["lanes_ms"] = round(s["now_ms"])
        if None not in t.values():
            return t
        page.wait_for_timeout(20)
    raise TimeoutError(f"timed out waiting for canvas/lanes: {t}")


def wheel_over(page, box, n: int, steps: list | None = None) -> None:
    page.mouse.move(box[0], box[1])
    for _ in range(n):
        page.mouse.wheel(0, -120)
        page.wait_for_timeout(150)
        if steps is not None:
            st = page.evaluate(JS_STATE)
            steps.append([st["wrapper_width"], st["scroll_left"]])


def center_of(page, selector_js: str) -> list:
    return page.evaluate(
        f"() => {{ const e = ({selector_js}); const r = e.getBoundingClientRect(); "
        "return [r.x + r.width / 2, r.y + r.height / 2]; }"
    )


def parse_readout(text: str | None) -> float | None:
    return float(text.split()[0]) if text else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    srv, port = serve(ROOT)
    console, page_errors = [], []
    summary: dict = {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--autoplay-policy=no-user-gesture-required"])
            page = browser.new_page(viewport={"width": 1100, "height": 500})
            page.on("console", lambda m: console.append(f"[{m.type}] {m.text}"))
            page.on("pageerror", lambda e: page_errors.append(str(e)))

            page.goto(f"http://127.0.0.1:{port}/tests/harness.html", wait_until="commit")
            timings = wait_timings(page)
            page.wait_for_timeout(500)

            s0 = page.evaluate(JS_STATE)
            page.screenshot(path=str(OUT / "01_initial.png"))
            page.screenshot(path=str(OUT / "05_functions.png"))  # same state: function polylines + row labels

            # 5. Decode swap: ~2 s after ready the decoded audio replaces the peaks rendering.
            page.wait_for_timeout(2000)
            s_dec = page.evaluate(JS_STATE)

            page.evaluate("window.wavlensModel.set('current_time', 12.0)")
            page.wait_for_timeout(500)
            s1 = page.evaluate(JS_STATE)
            page.screenshot(path=str(OUT / "02_seek.png"))

            # Zoom plugin listens for wheel on the scroll container; deltaY < 0 zooms in.
            wrapper_js = f"({JS_DEEP_QUERY})('.wrapper')[0]"
            box = center_of(page, wrapper_js)
            zoom_steps: list = []
            wheel_over(page, box, 6, zoom_steps)
            page.wait_for_timeout(700)
            s2 = page.evaluate(JS_STATE)
            page.screenshot(path=str(OUT / "03_zoom.png"))

            # "fft" toggle: light-DOM <button>; spectrogram plugin is built on first click.
            fft_btn = page.locator("#widget button", has_text="fft")
            fft_btn.click()
            try:
                page.wait_for_function(f"() => ({JS_STATE})().spec_canvases > 0", timeout=3_000)
            except Exception:
                pass
            page.wait_for_timeout(300)
            s3 = page.evaluate(JS_STATE)
            page.screenshot(path=str(OUT / "04_spectrogram.png"))

            # 1. Zoom while the spectrogram is shown: wheel over the spectrogram host.
            spec_js = f"(() => {{ const own = [...{wrapper_js}.children].filter((c) => !c.getAttribute('part') && c.tagName === 'DIV'); return own[own.length - 3]; }})()"
            spec_box = center_of(page, spec_js)
            page.mouse.move(spec_box[0], spec_box[1])
            fft_steps = []  # [wrapper scrollWidth, img clientWidth] after every wheel: must match, no lag.
            for _ in range(6):
                page.mouse.wheel(0, -120)
                page.wait_for_timeout(150)
                st = page.evaluate(JS_STATE)
                fft_steps.append([st["wrapper_scroll_width"], st["spec_client_width"]])
            page.wait_for_timeout(700)
            s4 = page.evaluate(JS_STATE)
            page.screenshot(path=str(OUT / "06_fft_zoom.png"))

            # 2. Deep zoom on the waveform view: toggle back, 12 more wheel events, read timeline labels.
            page.locator("#widget button", has_text="wave").click()
            page.wait_for_timeout(200)
            wheel_over(page, box, 12)
            page.wait_for_timeout(700)
            s5 = page.evaluate(JS_STATE)
            page.screenshot(path=str(OUT / "07_ticks.png"))

            # 3. Expanders: ▸ of group 1 -> expanded ["speech"]; ▸ of group 0 too -> both; ▾ of group 1 -> only group 0; close all.
            e_states = [page.evaluate(JS_STATE)]
            for idx in (1, 0, 1, 0):
                click_expander(page, idx)
                page.wait_for_timeout(300)
                e_states.append(page.evaluate(JS_STATE))
                if len(e_states) == 3:
                    page.screenshot(path=str(OUT / "08_expander.png"))
            click_expander(page, 0)   # leave group 0 open so column 1 exists for later geometry/label checks on this page
            page.wait_for_timeout(300)
            s6b = page.evaluate(JS_STATE)

            # 4. Transport: play (echoed to model), pause, then nudge back twice.
            buttons = page.locator("#widget button")
            back_btn, play_btn = buttons.nth(0), buttons.nth(1)
            play_btn.click()
            page.wait_for_timeout(600)
            s_play = page.evaluate(JS_STATE)
            play_btn.click()
            page.wait_for_timeout(300)
            s_pause_raw = page.evaluate(JS_STATE)
            # After a WebAudio pause the readout may lag the real position by one frame; a net-zero
            # nudge (▶ then ◀) refreshes it. The difference is reported as readout staleness.
            fwd_btn = buttons.nth(2)
            fwd_btn.click()
            page.wait_for_timeout(150)
            back_btn.click()
            page.wait_for_timeout(300)
            s_pause = page.evaluate(JS_STATE)
            back_btn.click()
            page.wait_for_timeout(200)
            back_btn.click()
            page.wait_for_timeout(300)
            s_nudge = page.evaluate(JS_STATE)

            # 7. Keys on the focused widget root: Space, ArrowRight x3, ArrowLeft. Count what reaches document.
            page.evaluate("() => { window.docKeys = 0; document.addEventListener('keydown', (e) => { if (['Space','ArrowLeft','ArrowRight'].includes(e.code)) window.docKeys++; }); }")
            readout_js = "[...document.querySelectorAll('#widget div')].find((d) => /monospace/.test(d.style.font))"
            rc = center_of(page, readout_js)
            page.mouse.click(rc[0], rc[1])
            page.wait_for_timeout(150)
            k0 = page.evaluate(JS_STATE)
            page.keyboard.press("Space")
            page.wait_for_timeout(500)
            k_play = page.evaluate(JS_STATE)
            page.keyboard.press("Space")
            page.wait_for_timeout(300)
            k_pause_raw = page.evaluate(JS_STATE)
            page.keyboard.press("ArrowRight")  # net-zero nudge refreshes the possibly stale readout
            page.wait_for_timeout(150)
            page.keyboard.press("ArrowLeft")
            page.wait_for_timeout(300)
            k_pause = page.evaluate(JS_STATE)
            for _ in range(3):
                page.keyboard.press("ArrowRight")
                page.wait_for_timeout(150)
            k_right = page.evaluate(JS_STATE)
            page.keyboard.press("ArrowLeft")
            page.wait_for_timeout(300)
            k_left = page.evaluate(JS_STATE)

            # 6. Mute: per-audio ☑ button, then the single all/none button (its label is the action), then Python side.
            mode_btn = page.locator("#widget button").filter(has_text=re.compile(r"^(all|none)$"))
            m0 = page.evaluate(JS_STATE)
            page.locator("#widget button", has_text="quiet").click()
            page.wait_for_timeout(200)
            m_quiet = page.evaluate(JS_STATE)
            mode_btn.click()  # reads "all" now (not every audio enabled)
            page.wait_for_timeout(200)
            m_all1 = page.evaluate(JS_STATE)
            mode_btn.click()  # reads "none"
            page.wait_for_timeout(200)
            m_none = page.evaluate(JS_STATE)
            mode_btn.click()  # reads "all"
            page.wait_for_timeout(200)
            m_all = page.evaluate(JS_STATE)
            page.evaluate("window.wavlensModel.set('enabled', [false, true])")
            page.wait_for_timeout(200)
            m_py = page.evaluate(JS_STATE)

            # 8 + 9. Variant page: row_height=24, height=60; then zoom 6 + pan 2000 px, labels must not move.
            page2 = browser.new_page(viewport={"width": 1100, "height": 500})
            page2.on("console", lambda m: console.append(f"[{m.type}] (rows) {m.text}"))
            page2.on("pageerror", lambda e: page_errors.append(f"(rows) {e}"))
            page2.goto(f"http://127.0.0.1:{port}/tests/harness.html?override="
                       + json.dumps({"row_height": 24, "height": 60}), wait_until="commit")
            wait_timings(page2)
            page2.wait_for_timeout(600)
            r0 = page2.evaluate(JS_STATE)
            page2.screenshot(path=str(OUT / "09_rows.png"))
            box2 = center_of(page2, wrapper_js)
            wheel_over(page2, box2, 6)
            page2.wait_for_timeout(500)
            page2.evaluate(f"() => {{ ({JS_DEEP_QUERY})('.scroll')[0].scrollLeft += 2000; }}")
            page2.wait_for_timeout(500)
            r1 = page2.evaluate(JS_STATE)
            page2.screenshot(path=str(OUT / "10_fixed_labels.png"))

            # 10 + 12. Playback stability and latency on a fresh page (decode finished first).
            page3 = browser.new_page(viewport={"width": 1100, "height": 500})
            page3.on("console", lambda m: console.append(f"[{m.type}] (play) {m.text}"))
            page3.on("pageerror", lambda e: page_errors.append(f"(play) {e}"))
            page3.goto(f"http://127.0.0.1:{port}/tests/harness.html", wait_until="commit")
            wait_timings(page3)
            page3.wait_for_timeout(2500)
            play3 = page3.locator("#widget button").nth(1)
            back3 = page3.locator("#widget button").nth(0)
            lat_play = page3.evaluate(JS_LATENCY, "play")          # 12a: click play -> first readout change
            samples_play = page3.evaluate(JS_SAMPLE, [2000, 50])     # 10a: 2 s of playback
            play3.click()                                            # pause
            page3.wait_for_timeout(100)
            p_state = page3.evaluate(JS_STATE)
            samples_pause = page3.evaluate(JS_SAMPLE, [1500, 50])    # 10b: pause holds
            play3.click()                                            # play again
            page3.wait_for_timeout(100)
            samples_replay = page3.evaluate(JS_SAMPLE, [1000, 50])   # 10c: ~1.0 s advance over 1 s
            page3.mouse.click(*center_of(page3, JS_READOUT))         # focus root
            page3.wait_for_timeout(50)
            space = page3.evaluate(JS_SPACE_PAUSE)                   # 10d: Space pauses within a frame (in-page timing)
            samples_space = page3.evaluate(JS_SAMPLE, [300, 50])
            lat_arrow = page3.evaluate(JS_LATENCY, "ArrowRight")     # 12b: keydown -> readout change (paused)
            page3.keyboard.press("Space")                            # play again
            page3.wait_for_timeout(300)
            back = page3.evaluate(JS_BACK_WHILE_PLAYING)             # 10e: ◀ while playing (in-page timing)
            samples_after_back = page3.evaluate(JS_SAMPLE, [400, 50])
            play3.click()

            # 11. Function rows: "clean" bound to audio row 0, "glob" on every audio row and lane group.
            page4 = browser.new_page(viewport={"width": 1100, "height": 500})
            page4.on("console", lambda m: console.append(f"[{m.type}] (fn) {m.text}"))
            page4.on("pageerror", lambda e: page_errors.append(f"(fn) {e}"))
            page4.goto(f"http://127.0.0.1:{port}/tests/harness.html?fixture=fixture_fn_rows.json", wait_until="commit")
            wait_timings(page4)
            page4.wait_for_timeout(600)
            f0 = page4.evaluate(JS_STATE)
            page4.screenshot(path=str(OUT / "11_fn_rows.png"))

            # 13-15. Legend stability, dim, solo on a fresh page (after the decode, so solo has a buffer).
            page5 = browser.new_page(viewport={"width": 1100, "height": 500})
            page5.on("console", lambda m: console.append(f"[{m.type}] (solo) {m.text}"))
            page5.on("pageerror", lambda e: page_errors.append(f"(solo) {e}"))
            page5.goto(f"http://127.0.0.1:{port}/tests/harness.html", wait_until="commit")
            wait_timings(page5)
            page5.wait_for_timeout(2500)
            page5.evaluate(JS_MARK_CANVASES)  # identity marks on the post-decode canvases
            l_states = [page5.evaluate(JS_STATE)]
            for _ in range(2):                                      # 13: open group 0, close it: column 0 width must not change
                click_expander(page5, 0)
                page5.wait_for_timeout(300)
                l_states.append(page5.evaluate(JS_STATE))
            click_expander(page5, 0)                                # open group 0 so its speaker labels exist for 14/15
            page5.wait_for_timeout(300)
            geo_state = page5.evaluate(JS_STATE)
            # 14. Dim is Python-only now: set model.dimmed and check label + segment opacity, then clear.
            page5.evaluate("window.wavlensModel.set('dimmed', ['l:0:speaker90'])"); page5.wait_for_timeout(200)
            dim_on = page5.evaluate(JS_STATE)
            page5.evaluate("window.wavlensModel.set('dimmed', [])"); page5.wait_for_timeout(200)
            dim_off = page5.evaluate(JS_STATE)
            # 15. Masks: a click on a name cycles neutral -> keep (solid outline) -> complement (dashed) -> neutral; keys intersect.
            spk = page5.locator('#widget div[data-key="l:0:speaker90"]')
            spk91 = page5.locator('#widget div[data-key="l:0:speaker91"]')
            grp = page5.locator('#widget div[data-key="g:0"]')
            W = 300
            m = {}
            spk.click(); page5.wait_for_timeout(W); m["spk90_keep"] = page5.evaluate(JS_STATE)
            spk.click(); page5.wait_for_timeout(W); m["spk90_complement"] = page5.evaluate(JS_STATE)
            spk.click(); page5.wait_for_timeout(W); m["spk90_neutral"] = page5.evaluate(JS_STATE)
            spk.click(); spk.click(); page5.wait_for_timeout(W)      # complement of speaker90 ...
            spk91.click(); page5.wait_for_timeout(W); m["not90_and_91"] = page5.evaluate(JS_STATE)   # ... intersected with speaker91
            spk.click(); spk91.click(); spk91.click(); page5.wait_for_timeout(W); m["reset"] = page5.evaluate(JS_STATE)
            grp.click(); page5.wait_for_timeout(W); m["group_keep"] = page5.evaluate(JS_STATE)
            grp.click(); page5.wait_for_timeout(W); m["group_complement"] = page5.evaluate(JS_STATE)
            grp.click(); page5.wait_for_timeout(W); m["group_neutral"] = page5.evaluate(JS_STATE)
            # 16. Mute white-out: toggle "quiet" off -> .mute over row 1; on -> none.
            quiet = page5.locator("#widget button", has_text="quiet")
            quiet.click(); page5.wait_for_timeout(W); mute_on = page5.evaluate(JS_STATE)
            quiet.click(); page5.wait_for_timeout(W); mute_off = page5.evaluate(JS_STATE)
            # 17. Playback under a complement mask (everything but speaker90): start at 6.0 s, the 6.69-7.12 segment must be skipped.
            spk.click(); spk.click(); page5.wait_for_timeout(W)
            m["spk90_complement_again"] = page5.evaluate(JS_STATE)
            page5.evaluate("window.wavlensModel.set('current_time', 6.0)"); page5.wait_for_timeout(150)
            page5.locator("#widget button").nth(1).click()
            so_play = page5.evaluate(JS_SAMPLE, [1500, 50])
            page5.locator("#widget button").nth(1).click()
            page5.wait_for_timeout(150)
            page5.screenshot(path=str(OUT / "15_solo.png"))
            spk.click(); page5.wait_for_timeout(W)
            so_off = page5.evaluate(JS_STATE)

            # 16. Cleanup while playing: AudioContext closes, no more timeupdate -> model sets.
            page6 = browser.new_page(viewport={"width": 1100, "height": 500})
            page6.on("console", lambda m: console.append(f"[{m.type}] (cleanup) {m.text}"))
            page6.on("pageerror", lambda e: page_errors.append(f"(cleanup) {e}"))
            page6.goto(f"http://127.0.0.1:{port}/tests/harness.html", wait_until="commit")
            wait_timings(page6)
            page6.wait_for_timeout(2500)
            page6.locator("#widget button").nth(1).click()
            page6.wait_for_timeout(400)
            c0 = page6.evaluate(JS_VIEWS)
            c_call = page6.evaluate("() => { const t0 = performance.now(); try { window.wavlensCleanup(); return { ok: true, ms: Math.round(performance.now() - t0) }; } catch (e) { return { ok: false, error: String(e && e.stack || e) }; } }")
            page6.wait_for_timeout(100)
            c1 = page6.evaluate(JS_VIEWS)
            page6.wait_for_timeout(600)
            c2 = page6.evaluate(JS_VIEWS)
            c_dom = page6.evaluate("() => ({ widget_children: document.getElementById('widget').children.length, buttons: document.querySelectorAll('#widget button').length })")

            # 18. Drag-to-reorder on a 3-group fixture: solo a/speaker90, drag "c" above "a", keys remapped; 2 px move = click.
            page8 = browser.new_page(viewport={"width": 1100, "height": 500})
            page8.on("console", lambda m: console.append(f"[{m.type}] (drag) {m.text}"))
            page8.on("pageerror", lambda e: page_errors.append(f"(drag) {e}"))
            page8.goto(f"http://127.0.0.1:{port}/tests/harness.html?fixture=fixture_3groups.json", wait_until="commit")
            wait_timings(page8)
            page8.wait_for_timeout(600)
            click_expander(page8, 0)                                 # open "a"
            page8.wait_for_timeout(300)
            page8.locator('#widget div[data-key="l:0:speaker90"]').click()
            page8.wait_for_timeout(300)
            g0 = page8.evaluate(JS_STATE)
            # Drag "c" (group 2) above "a" (group 0): pointer down on its name, move > 5 px in steps, release just below the lanes top.
            c_name = next(n for n in g0["name_divs"] if n["text"] == "c")
            a_name = next(n for n in g0["name_divs"] if n["text"] == "a")
            page8.mouse.move(c_name["x"] + 5, c_name["y"] + c_name["h"] / 2)
            page8.mouse.down()
            page8.mouse.move(c_name["x"] + 5, a_name["y"] + 1, steps=12)
            page8.wait_for_timeout(100)
            g_mid = page8.evaluate("() => { const m = [...document.querySelectorAll('#widget div')].find((d) => /2px solid/.test(d.style.borderTop)); return m ? { display: m.style.display, top: m.style.top } : null; }")
            page8.mouse.up()
            page8.wait_for_timeout(400)
            g1 = page8.evaluate(JS_STATE)
            page8.screenshot(path=str(OUT / "18_drag.png"))
            # A 2 px move must not reorder and must still cycle the mask of the clicked name.
            a_now = next(n for n in g1["name_divs"] if n["text"] == "a")
            page8.mouse.move(a_now["x"] + 5, a_now["y"] + a_now["h"] / 2)
            page8.mouse.down()
            page8.mouse.move(a_now["x"] + 5, a_now["y"] + a_now["h"] / 2 + 2, steps=2)
            page8.mouse.up()
            page8.wait_for_timeout(400)
            g2 = page8.evaluate(JS_STATE)

            # 17. Two views of one model: play in A, B follows current_time; pause in A -> both not playing.
            page7 = browser.new_page(viewport={"width": 1100, "height": 900})
            page7.on("console", lambda m: console.append(f"[{m.type}] (views) {m.text}"))
            page7.on("pageerror", lambda e: page_errors.append(f"(views) {e}"))
            page7.goto(f"http://127.0.0.1:{port}/tests/harness.html?views=2", wait_until="commit")
            page7.wait_for_function("window.wavlensReady || window.wavlensError", timeout=60_000)
            page7.wait_for_timeout(3000)
            v0 = page7.evaluate(JS_VIEWS)
            page7.locator("#widget button").nth(1).click()           # play in view A
            page7.wait_for_timeout(600)
            v1 = page7.evaluate(JS_VIEWS)
            # Readout cadence per view: A updates every frame (own playback), B only via model.current_time (10 Hz).
            cadence = page7.evaluate("""async () => {
              const ros = [...document.querySelectorAll('[id^="widget"]')].map((h) => [...h.querySelectorAll('div')].find((d) => /monospace/.test(d.style.font)));
              const seen = ros.map(() => new Set());
              const t0 = performance.now();
              while (performance.now() - t0 < 400) { ros.forEach((r, i) => seen[i].add(r.textContent)); await new Promise((r) => setTimeout(r, 20)); }
              return seen.map((s) => s.size);
            }""")
            v2 = page7.evaluate(JS_VIEWS)
            page7.locator("#widget button").nth(1).click()           # pause in view A
            page7.wait_for_timeout(300)
            v3 = page7.evaluate(JS_VIEWS)
            page7.screenshot(path=str(OUT / "17_two_views.png"))

            browser.close()

        t_before, t_after = parse_readout(s_pause["readout"]), parse_readout(s_nudge["readout"])
        summary = {
            "lane_segments": s0["lane_segments"],
            "lane_groups": s0["lane_groups"],
            "fn_polylines": {"audio_rows": s0["fn_polylines_audio"], "lane_groups": s0["fn_polylines_lanes"],
                             "total": s0["fn_polylines_audio"] + s0["fn_polylines_lanes"]},
            "row_labels": s0["row_labels"],
            "buttons": s0["buttons"],
            "load_ms_to_waveform_canvas": timings["canvas_ms"],
            "load_ms_to_lane_segments": timings["lanes_ms"],
            "wrapper_width_initial": s0["wrapper_width"],
            "wrapper_width_after_zoom": s2["wrapper_width"],
            "wrapper_height": s0["wrapper_height"],
            "current_time_after_seek": s1["current_time"],
            "scroll_left_after_seek": s1["scroll_left"],
            "scroll_left_after_zoom": s2["scroll_left"],
            "zoom_steps_width_scroll": zoom_steps,
            "pointer_xy": box,
            "rows_scroll_width_after_zoom": {
                "wrapper": s2["wrapper_scroll_width"],
                "timeline": s2["timeline_scroll_width"],
                "lanes": s2["lanes_scroll_width"],
            },
            "rows_same_resolution": s2["wrapper_scroll_width"] == s2["timeline_scroll_width"] == s2["lanes_scroll_width"],
            "fft": {
                "spectrogram_visible": s3["spec_visible"],
                "spectrogram_kind": s3["spec_kind"],
                "spectrogram_canvases": s3["spec_canvases"],
                "waveform_hidden": s3["waveform_hidden"],
            },
            "fft_zoom": {
                "img_visible_after_fft_click": s3["spec_img_visible"],
                "wrapper_width_before": s3["wrapper_width"],
                "wrapper_width_after": s4["wrapper_width"],
                "img_client_width_before": s3["spec_client_width"],
                "img_client_width_after": s4["spec_client_width"],
                "steps_wrapper_scroll_width_vs_img_client_width": fft_steps,
                "img_matches_wrapper_every_step": all(w == i for w, i in fft_steps),
            },
            "deep_zoom_ticks": {
                "wrapper_width": s5["wrapper_width"],
                "px_per_sec": round(s5["wrapper_width"] / window_duration(s5), 1) if s5["wrapper_width"] else None,
                "notches": s5["timeline_notches"],
                "label_count": len(s5["timeline_labels"]),
                "first_10_labels": s5["timeline_labels"][:10],
                "notches_at_full_view": s0["timeline_notches"],
                "labels_at_full_view": s0["timeline_labels"][:10],
            },
            "expander": expander_summary(e_states, s6b),
            "transport": {
                "playing_after_play_click": s_play["playing"],
                "play_button_text_while_playing": s_play["play_button"],
                "play_button_text_after_pause": s_pause["play_button"],
                "readout_while_playing": s_play["readout"],
                "playing_after_pause_click": s_pause_raw["playing"],
                "readout_right_after_pause": s_pause_raw["readout"],
                "readout_stale_after_pause_ms": round(1000 * (parse_readout(s_pause["readout"]) - parse_readout(s_pause_raw["readout"]))),
                "readout_before_nudge": s_pause["readout"],
                "readout_after_2x_back": s_nudge["readout"],
                "readout_delta_s": round(t_after - t_before, 3) if None not in (t_before, t_after) else None,
                "model_current_time_after_nudge": s_nudge["current_time"],
                # model.current_time is throttled to 10 Hz, so compare the post-nudge echo with the pre-nudge readout.
                "model_delta_vs_readout_before_s": round(s_nudge["current_time"] - t_before, 3) if t_before is not None else None,
            },
            "keys": {
                "playing_after_space": k_play["playing"],
                "playing_after_second_space": k_pause_raw["playing"],
                "readout_right_after_pause": k_pause_raw["readout"],
                "readout_stale_after_pause_ms": round(1000 * (parse_readout(k_pause["readout"]) - parse_readout(k_pause_raw["readout"]))),
                "readout_before_arrows": k_pause["readout"],
                "readout_after_3x_right": k_right["readout"],
                "right_delta_s": round(parse_readout(k_right["readout"]) - parse_readout(k_pause["readout"]), 3),
                "readout_after_left": k_left["readout"],
                "left_delta_s": round(parse_readout(k_left["readout"]) - parse_readout(k_right["readout"]), 3),
                "scroll_y_before_after": [k0["scroll_y"], k_left["scroll_y"]],
                "page_scrollable": True,
                "keydowns_reaching_document": k_left["doc_keys"],
            },
            "mute": {
                "mode_button_initial": m0["mode_button"],
                "after_quiet_click": {"enabled": m_quiet["enabled"], "toggles": m_quiet["toggles"], "mode_button": m_quiet["mode_button"]},
                "after_all_click": {"enabled": m_all1["enabled"], "mode_button": m_all1["mode_button"]},
                "after_none_click": {"enabled": m_none["enabled"], "toggles": [t["text"] for t in m_none["toggles"]], "mode_button": m_none["mode_button"]},
                "after_all_click_2": {"enabled": m_all["enabled"], "toggles": [t["text"] for t in m_all["toggles"]], "mode_button": m_all["mode_button"]},
                "after_python_set_false_true": {"enabled": m_py["enabled"], "toggles": m_py["toggles"], "mode_button": m_py["mode_button"]},
                "mode_button_sequence_ok": [m0["mode_button"], m_quiet["mode_button"], m_all1["mode_button"], m_none["mode_button"], m_all["mode_button"], m_py["mode_button"]]
                    == ["none", "all", "none", "all", "none", "all"],
            },
            "row_heights": {
                "override": {"row_height": 24, "height": 60},
                "lane_rows": r0["lane_rows"],
                "lanes_height": r0["lanes_height"],
                "lanes_height_ok": r0["lanes_height"] == r0["lane_rows"] * 24,
                "canvases_height": r0["canvases_height"],
                "channel_heights": r0["channel_heights"],
                "audio_rows_note": "2 audios + 1 mix row (mean of the enabled audios) = 3 audio rows",
                "audio_rows_ok": r0["channel_heights"] == [60] * len(r0["channel_heights"]) and len(r0["channel_heights"]) == 3,
                "wrapper_height": r0["wrapper_height"],
            },
            "fixed_labels": {
                "wrapper_width_after_zoom": r1["wrapper_width"],
                "scroll_left_after_pan": r1["scroll_left"],
                "toggle_x_before_after": [[t["x"] for t in r0["toggles"]], [t["x"] for t in r1["toggles"]]],
                "label_x_before_after": [r0["label_xs"], r1["label_xs"]],
                "gutter_children": r0["gutter_children"],
                "gutter_xs_before_after": [r0["gutter_xs"], r1["gutter_xs"]],
                "gutter_widths_before_after": [r0["gutter_widths"], r1["gutter_widths"]],
                "all_fixed": [t["x"] for t in r0["toggles"]] == [t["x"] for t in r1["toggles"]]
                    and r0["label_xs"] == r1["label_xs"] and r0["gutter_xs"] == r1["gutter_xs"] and r0["gutter_children"] > 0,
            },
            "playback": playback_summary(samples_play, samples_pause, samples_replay, p_state, space,
                                         samples_space, back, samples_after_back),
            "latency_ms": {"play_click_to_readout": lat_play, "arrow_keydown_to_readout": lat_arrow,
                           "both_under_50ms": 0 <= lat_play < 50 and 0 <= lat_arrow < 50},
            "function_rows": {
                "functions": [(f["name"], f.get("row")) for f in json.load(open(ROOT / "tests" / "fixture_fn_rows.json"))["functions"]],
                # Each function is drawn as a white halo polyline plus the colored one; compare distinct names.
                "per_audio_row": [sorted(set(r)) for r in f0["fn_per_audio_row"]],
                "per_lane_group": [sorted(set(r)) for r in f0["fn_per_lane_group"]],
                "polylines_per_function": (len(f0["fn_per_audio_row"][0]) // max(1, len(set(f0["fn_per_audio_row"][0])))) if f0["fn_per_audio_row"] else None,
                # With >= 2 audios a third "mix" row (mean of the enabled audios) is drawn; row -1 functions appear on it too.
                "audio_rows_drawn": len(f0["fn_per_audio_row"]),
                "ok": [sorted(set(r)) for r in f0["fn_per_audio_row"]] == [["clean", "glob"], ["glob"], ["glob"]]
                      and [sorted(set(r)) for r in f0["fn_per_lane_group"]] == [["glob"]],
            },
            "transport_buttons": {"titles": s0["transport_titles"], "svg_paths": s0["transport_have_svg_path"]},
            "expander_stability": {
                "expanded_sequence": [s["expanded"] for s in l_states],
                "col0_widths": [s["gutter_widths"][0] for s in l_states],
                "col1_widths": [s["gutter_widths"][1] for s in l_states],
                "col1_display": [s["col1_display"] for s in l_states],
                "wrapper_scroll_widths": [s["wrapper_scroll_width"] for s in l_states],
                "canvases_same_objects_per_state": [s["marked_canvases"] == s["waveform_canvases"] and s["waveform_canvases"] > 0 for s in l_states],
                "col0_width_unchanged": len({s["gutter_widths"][0] for s in l_states}) == 1,
                "ok": len({s["gutter_widths"][0] for s in l_states}) == 1 and [s["col1_display"] for s in l_states] == ["none", "block", "none"],
            },
            "drag_reorder": drag_summary(g0, g_mid, g1, g2),
            "dim_python": {
                "set": {"label_opacity": label_of(dim_on, "l:0:speaker90")["opacity"], "seg_opacity_speaker90": seg_opacity(dim_on, "0", "speaker90"),
                        "seg_opacity_speaker91": seg_opacity(dim_on, "0", "speaker91")},
                "cleared": {"label_opacity": label_of(dim_off, "l:0:speaker90")["opacity"], "seg_opacity_speaker90": seg_opacity(dim_off, "0", "speaker90")},
                "ok": label_of(dim_on, "l:0:speaker90")["opacity"] == "0.25" and seg_opacity(dim_on, "0", "speaker90") == ["0.25"]
                      and seg_opacity(dim_on, "0", "speaker91") == [""] and seg_opacity(dim_off, "0", "speaker90") == [""],
            },
            "masks": mask_summary(m, so_off, fixture_groups()),
            "mute_whiteout": {
                "toggle_texts": [t["text"] for t in mute_on["toggles"]],
                "after_mute_quiet": {"enabled": mute_on["enabled"], "mute_rows": mute_on["mute_rows"], "quiet_checked": mute_on["toggles"][1]["checked"]},
                "after_unmute": {"enabled": mute_off["enabled"], "mute_rows": mute_off["mute_rows"], "quiet_checked": mute_off["toggles"][1]["checked"]},
                "ok": mute_on["mute_rows"] == ["1"] and mute_off["mute_rows"] == [] and mute_on["enabled"] == [True, False] and mute_off["enabled"] == [True, True],
            },
            "complement_playback": complement_playback_summary(so_play, fixture_groups()),
            "column_geometry": geometry_summary(geo_state, fixture_groups(), fixture_row_height()),
            "cleanup": {
                "before": {"contexts": c0["contexts"], "playing": c0["playing"], "readout": c0["views"][0]["readout"], "log_len": c0["log_len"]},
                "call": c_call,
                "after_100ms": {"contexts": c1["contexts"], "log_len": c1["log_len"]},
                "after_700ms": {"contexts": c2["contexts"], "log_len": c2["log_len"], "playing": c2["playing"]},
                "dom_after": c_dom,
                "context_closed_within_100ms": bool(c1["contexts"]) and c1["contexts"][-1] == "closed",
                "no_model_sets_after_cleanup": c2["log_len"] == c1["log_len"],
                "ok": c_call.get("ok", False) and bool(c1["contexts"]) and c1["contexts"][-1] == "closed" and c2["log_len"] == c1["log_len"],
            },
            "two_views": {
                "initial": v0,
                "after_play_in_A_600ms": v1,
                "after_play_in_A_1200ms": v2,
                "after_pause_in_A": v3,
                "B_readout_follows": v2["views"][1]["readout"] != v0["views"][1]["readout"]
                    and abs(parse_readout(v2["views"][1]["readout"]) - parse_readout(v2["views"][0]["readout"])) < 0.3,
                "distinct_readouts_in_400ms_A_B": cadence,
                "B_not_playing_itself": v1["views"][1]["play_icon"] == "play" and cadence[1] <= 6 and cadence[0] >= 12,
                "context_states_note": "headless runs with autoplay allowed, so every AudioContext reports 'running' from creation",
                "only_A_running_by_context_state": v1["contexts"][0] == "running" and v1["contexts"][1] != "running",
                "both_not_playing_after_pause": v3["playing"] is False and all(v["play_icon"] == "play" for v in v3["views"]),
            },
            "decode_swap": {
                "canvases_at_first_paint": timings.get("canvas_ms") is not None,
                "canvas_count_first_paint_vs_2s": [s0["waveform_canvases"], s_dec["waveform_canvases"]],
                "canvas_widths_2s": s_dec["waveform_canvas_widths"],
                "first_paint_canvases_still_present": s_dec["marked_canvases"],
                "re_rendered": s_dec["marked_canvases"] == 0 and s_dec["waveform_canvases"] > 0,
                "error_after_2s": s_dec["error"],
            },
            "model_log_tail": s_nudge["log_tail"],
            "console_errors": [c for c in console if c.startswith("[error]")],
            "page_errors": page_errors,
            "screenshots": [str(OUT / n) for n in
                            ("01_initial.png", "02_seek.png", "03_zoom.png", "04_spectrogram.png",
                             "05_functions.png", "06_fft_zoom.png", "07_ticks.png", "08_expander.png",
                             "09_rows.png", "10_fixed_labels.png", "11_fn_rows.png", "15_solo.png", "17_two_views.png",
                             "18_drag.png")],
        }
    finally:
        srv.shutdown()
        srv.server_close()
    print(json.dumps(summary, indent=2))


def label_of(state: dict, key: str) -> dict:
    return next((l for l in state["name_labels"] if l["key"] == key), {"opacity": None, "outline": None})


def seg_opacity(state: dict, group: str, label: str | None = None) -> list:
    return sorted({s["opacity"] for s in state["seg_styles"] if s["group"] == group and (label is None or s["label"] == label)})


def fixture_groups() -> list:
    return json.load(open(ROOT / "tests" / "fixture.json"))["groups"]


def fixture_row_height() -> int:
    return json.load(open(ROOT / "tests" / "fixture.json")).get("row_height", 16)


def union(intervals: list) -> list:
    out: list = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def click_expander(page, idx: int) -> None:
    """Click the ▸/▾ button of lane group `idx` (column 0, in group order)."""
    page.evaluate("""(i) => { const col0 = document.querySelector('#widget > div > div:nth-child(3) > div');
      const b = [...col0.querySelectorAll('button')].filter((b) => /[▸▾]/.test(b.textContent))[i]; b.click(); }""", idx)


def expander_summary(states: list, final: dict) -> dict:
    steps = ["start", "open_1", "open_0_too", "close_1", "close_0"]
    out = {}
    for name, s in zip(steps, states):
        out[name] = {"expanded": s["expanded"], "expander_texts": [e["text"] for e in s["expanders"]],
                     "col1_display": s["col1_display"], "col1_labels": [l["key"] for l in s["col1_labels"]],
                     "col0_width": s["gutter_widths"][0], "col1_width": s["gutter_widths"][1],
                     "segments_with_text": s["lane_segments_with_text"]}
    exp = [s["expanded"] for s in states]
    labels = [[l["key"] for l in s["col1_labels"]] for s in states]
    ok = (exp == [[], ["speech"], ["speech", "reference"], ["reference"], []]
          and labels[1] == [] and sorted(labels[2]) == ["l:0:speaker90", "l:0:speaker91"] and sorted(labels[3]) == ["l:0:speaker90", "l:0:speaker91"]
          and states[4]["col1_display"] == "none" and states[2]["col1_display"] != "none"
          and len({s["gutter_widths"][0] for s in states}) == 1
          and all(s["lane_segments_with_text"] == states[0]["lane_segments_with_text"] for s in states))
    out["note"] = "group 1 ('speech') has a blank-label row, so opening it alone adds no column-1 label and column 1 stays display:none"
    out["final_open_0"] = {"expanded": final["expanded"], "col1_labels": [l["key"] for l in final["col1_labels"]]}
    out["ok"] = ok
    return out


def drag_summary(g0: dict, mid: dict | None, g1: dict, g2: dict) -> dict:
    return {
        "before": {"group_names": g0["group_names"], "solo": g0["solo"], "seg_top_by_group": g0["seg_top_by_group"],
                   "outlined": [n["key"] for n in g0["name_labels"] if n["outline"]]},
        "marker_while_dragging": mid,
        "after_drag_c_above_a": {"group_names": g1["group_names"], "solo": g1["solo"], "expanded": g1["expanded"],
                                 "seg_top_by_group": g1["seg_top_by_group"],
                                 "outlined": [n["key"] for n in g1["name_labels"] if n["outline"]],
                                 "col1_labels": [l["key"] for l in g1["col1_labels"]]},
        "after_2px_move_on_a": {"group_names": g2["group_names"], "solo": g2["solo"],
                                "outlined": [n["key"] for n in g2["name_labels"] if n["outline"]]},
        "ok": g0["group_names"] == ["a", "b", "c"] and g0["solo"] == ["l:0:speaker90"]
              and g1["group_names"] == ["c", "a", "b"] and g1["solo"] == ["l:1:speaker90"]
              and "l:1:speaker90" in [n["key"] for n in g1["name_labels"] if n["outline"]]
              and g1["seg_top_by_group"].get("0", 1e9) < g1["seg_top_by_group"].get("1", 1e9) < g1["seg_top_by_group"].get("2", 1e9)
              and mid is not None and mid["display"] != "none"
              and g2["group_names"] == ["c", "a", "b"] and sorted(g2["solo"]) == sorted(["l:1:speaker90", "g:1"]),
    }


def legend_stability_summary(states: list) -> dict:
    keys = ("wrapper_scroll_width", "gutter_widths", "waveform_canvases", "waveform_canvas_widths")
    snap = [{k: s[k] for k in keys} for s in states]
    side_both_same = snap[1] == snap[2]
    inline_same = snap[0] == snap[3]
    return {
        "legend_sequence": [s["legend"] for s in states],
        "button_sequence": [s["legend_button"] for s in states],
        "snapshots": snap,
        "unchanged_all": all(x == snap[0] for x in snap[1:]),
        "side_equals_both": side_both_same,
        "inline_restored": inline_same,
        "canvases_same_objects_per_state": [s["marked_canvases"] == s["waveform_canvases"] and s["waveform_canvases"] > 0 for s in states],
        "col1_display_widths": [s["gutter_widths"][1] for s in states],
        "segments_transparent": [s["lane_segments_transparent"] for s in states],
    }


def geometry_summary(state: dict, groups: list, rh: int) -> dict:
    rows_before = 0
    expected = {}
    for gi, g in enumerate(groups):
        for i, r in enumerate(g["rows"]):
            expected[f"l:{gi}:{r}"] = state["lanes_top"] + (rows_before + i) * rh + rh / 2
        expected[f"g:{gi}"] = state["lanes_top"] + rows_before * rh + rh / 2
        rows_before += len(g["rows"])
    checks = []
    for l in state["name_labels"]:
        if l["key"] not in expected:
            continue
        checks.append({"key": l["key"], "top": round(l["top"], 1), "center": round(l["center"], 1),
                       "expected_center": round(expected[l["key"]], 1), "delta": round(l["center"] - expected[l["key"]], 1),
                       "below_timeline": l["top"] >= state["timeline_bottom"]})
    return {
        "timeline_bottom": state["timeline_bottom"], "lanes_top": state["lanes_top"], "row_height": rh,
        "labels": checks,
        "ok": bool(checks) and all(c["below_timeline"] and abs(c["delta"]) <= 2 for c in checks),
    }


def intersect(a: list, b: list) -> list:
    out = []
    for s1, e1 in a:
        for s2, e2 in b:
            s, e = max(s1, s2), min(e1, e2)
            if e > s:
                out.append([s, e])
    return union(out)


def close_lists(a: list, b: list, tol: float = 0.01) -> bool:
    return len(a) == len(b) and all(abs(x[0] - y[0]) <= tol and abs(x[1] - y[1]) <= tol for x, y in zip(a, b))


def mask_summary(m: dict, off: dict, groups: list) -> dict:
    D = float(json.load(open(ROOT / "tests" / "fixture.json"))["duration"])
    segs = groups[0]["segments"]
    s90 = union([[s["start"], s["end"]] for s in segs if s["label"] == "speaker90"])
    s91 = union([[s["start"], s["end"]] for s in segs if s["label"] == "speaker91"])
    grp = union([[s["start"], s["end"]] for s in segs])
    comp = lambda k: gaps_of(k, D)
    expect = {
        # Outline style as the browser serializes it: "rgb(238, 0, 0) solid 1px" / "... dashed 1px".
        "spk90_keep": (["l:0:speaker90"], comp(s90), "solid"),
        "spk90_complement": (["!l:0:speaker90"], s90, "dashed"),            # white-outs = exactly the speaker90 segments
        "spk90_neutral": ([], [], ""),
        "not90_and_91": (["!l:0:speaker90", "l:0:speaker91"], comp(intersect(comp(s90), s91)), None),
        "reset": ([], [], ""),
        "group_keep": (["g:0"], comp(grp), "solid"),
        "group_complement": (["!g:0"], grp, "dashed"),
        "group_neutral": ([], [], ""),
        "spk90_complement_again": (["!l:0:speaker90"], s90, "dashed"),
    }
    out, ok = {}, True
    for name, st in m.items():
        keys, gaps, outline = expect[name]
        key = "g:0" if name.startswith("group") else "l:0:speaker90"
        got_outline = label_of(st, key)["outline"]
        res = {"solo": st["solo"], "solo_expected": keys, "gaps": st["gaps_s"], "gaps_expected": [[round(a, 3), round(b, 3)] for a, b in gaps],
               "gaps_match": close_lists(st["gaps_s"], gaps), "outline": got_outline,
               "outline_ok": outline is None or (outline == "" and not got_outline) or (outline != "" and outline in got_outline)}
        res["ok"] = sorted(st["solo"] or []) == sorted(keys) and res["gaps_match"] and res["outline_ok"]
        ok = ok and res["ok"]
        out[name] = res
    out["after_all_neutral"] = {"solo": off["solo"], "gaps": off["gaps_s"], "any_outline": any(l["outline"] for l in off["name_labels"])}
    out["ok"] = ok and off["solo"] == [] and off["gaps_s"] == [] and not any(l["outline"] for l in off["name_labels"])
    return out


def complement_playback_summary(play: list, groups: list) -> dict:
    s90 = union([[s["start"], s["end"]] for s in groups[0]["segments"] if s["label"] == "speaker90"])
    ro = [s[1] for s in play]
    incs = [round(b - a, 3) for a, b in zip(ro, ro[1:])]
    TOL = 0.05
    inside = [t for t in ro if any(a + TOL < t < b - TOL for a, b in s90)]
    first = next((seg for seg in s90 if seg[0] >= 6.0), None)
    return {"readout_first_last": [ro[0], ro[-1]], "samples": len(ro), "monotonic": all(i >= 0 for i in incs),
            "max_increment_s": max(incs) if incs else None, "skipped_segment": first,
            "samples_inside_speaker90": inside,
            "ok": not inside and first is not None and max(incs) >= round(first[1] - first[0], 3) - 0.06 and all(i >= 0 for i in incs)}


def gaps_of(kept: list, duration: float) -> list:
    out, prev = [], 0.0
    for s, e in kept + [[duration, duration]]:
        if s > prev:
            out.append([round(prev, 3), round(s, 3)])
        prev = max(prev, e)
    return out


def solo_summary(on: dict, seek: dict, play: list, two: dict, off: dict, group: dict, groups: list,
                 one: dict, zero_before: dict, zero_300: dict, zero_play: list) -> dict:
    """Skip-gaps semantics: masked 30 s buffer, white-outs over the gaps, playhead skips them; axis unchanged."""
    duration = float(json.load(open(ROOT / "tests" / "fixture.json"))["duration"])
    segs = groups[0]["segments"]
    kept90 = union([[s["start"], s["end"]] for s in segs if s["label"] == "speaker90"])
    kept91 = union([[s["start"], s["end"]] for s in segs if s["label"] == "speaker91"])
    kept_both = union([[s["start"], s["end"]] for s in segs if s["label"] in ("speaker90", "speaker91")])
    kept_group = union([[s["start"], s["end"]] for s in segs])
    buf_on = on["buffers"][-1]["duration"] if on["buffers"] else None
    ro = [s[1] for s in play]
    incs = [round(b - a, 3) for a, b in zip(ro, ro[1:])]
    first_gap = next((g for g in gaps_of(kept90, duration) if g[0] > 6.9), None)
    TOL = 0.05  # one frame of readout lag (~16 ms) plus sampling slack
    in_kept = lambda t: any(a - TOL <= t <= b + TOL for a, b in kept90)
    zr = [s[1] for s in zero_play]
    outside = [t for t in zr if not in_kept(t)]
    return {
        "play_from_zero": {
            "readout_after_seek_to_0": zero_before["readout"],
            "readout_300ms_after_play": zero_300["readout"],
            "jumped_to_first_segment_within_300ms": parse_readout(zero_300["readout"]) >= kept90[0][0],
            "samples": len(zr), "readout_first_last": [zr[0], zr[-1]],
            "monotonic": all(b >= a for a, b in zip(zr, zr[1:])),
            "samples_outside_kept": outside, "never_in_gap": not outside,
        },
        "solo_set": {
            "after_speaker90": on["solo"],
            "after_plus_speaker91": {"solo": two["solo"], "outlined": sorted(l["key"] for l in two["name_labels"] if l["outline"]),
                                     "gap_whiteouts": two["gap_whiteouts"], "gaps_expected_union": len(gaps_of(kept_both, duration))},
            "after_minus_speaker90": {"solo": one["solo"], "outlined": sorted(l["key"] for l in one["name_labels"] if l["outline"]),
                                      "gap_whiteouts": one["gap_whiteouts"], "gaps_expected_speaker91": len(gaps_of(kept91, duration))},
            "after_plus_group": {"solo": group["solo"], "outlined": sorted(l["key"] for l in group["name_labels"] if l["outline"]),
                                 "gap_whiteouts": group["gap_whiteouts"], "gaps_expected_group": len(gaps_of(kept_group, duration))},
            "after_all_off": {"solo": off["solo"], "gap_whiteouts": off["gap_whiteouts"]},
            "ok": len(on["solo"] or []) == 1 and len(two["solo"]) == 2 and len(one["solo"]) == 1 and one["solo"] == ["l:0:speaker91"]
                  and len(group["solo"]) == 2 and off["solo"] == [] and one["gap_whiteouts"] == len(gaps_of(kept91, duration)),
        },
        "speaker90_kept_pieces": kept90, "speaker90_gaps_expected": gaps_of(kept90, duration),
        "on": {"buffers_created_on_solo": len(on["buffers"]), "buffer_created_duration_s": round(buf_on, 3) if buf_on else None,
               "gain_automation_no_new_buffer": len(on["buffers"]) == 0,
               "gap_whiteouts": on["gap_whiteouts"], "gap_whiteouts_expected": len(gaps_of(kept90, duration)),
               "join_marks": on["join_marks"], "segment_divs_total": on["lane_segments"], "spec_tiles": on["spec_tiles"],
               "last_ruler_label": on["last_ruler_label"], "axis_unchanged": on["last_ruler_label"] == off["last_ruler_label"],
               "label_outline": label_of(on, "l:0:speaker90")["outline"],
               "other_labels_outlined": [l["key"] for l in on["name_labels"] if l["outline"] and l["key"] != "l:0:speaker90"]},
        "seek_click": {u: {**v, "in_gap": not any(a <= float(u) <= b for a, b in kept90),
                           "expected_readout_s": (next((a for a, b in kept90 if a > float(u)), None) if not any(a <= float(u) <= b for a, b in kept90) else float(u))}
                       for u, v in seek.items()},
        "play_1s_from_6.9": {"readout_first_last": [ro[0], ro[-1]], "advance_s": round(ro[-1] - ro[0], 3), "monotonic": all(i >= 0 for i in incs),
                             "max_increment_s": max(incs) if incs else None, "gap_crossed": first_gap,
                             "gap_skipped": first_gap is not None and max(incs) >= round(first_gap[1] - first_gap[0], 3) - 0.06},
        "off": {"gap_whiteouts": off["gap_whiteouts"], "any_outline": any(l["outline"] for l in off["name_labels"]),
                "segment_divs_total": off["lane_segments"], "last_ruler_label": off["last_ruler_label"]},
        "ok": on["gap_whiteouts"] == len(gaps_of(kept90, duration)) and bool(label_of(on, "l:0:speaker90")["outline"])
              and len(on["buffers"]) == 0 and on["lane_segments"] == off["lane_segments"]
              and two["gap_whiteouts"] == len(gaps_of(kept_both, duration)) and off["gap_whiteouts"] == 0
              and not any(l["outline"] for l in off["name_labels"]) and group["gap_whiteouts"] == len(gaps_of(kept_group, duration))
              and not outside and parse_readout(zero_300["readout"]) >= kept90[0][0],
    }


def dim_summary(d: dict) -> dict:
    out = {}
    for name, st in d.items():
        out[name] = {
            "speaker90_label_opacity": label_of(st, "l:0:speaker90")["opacity"],
            "reference_label_opacity": label_of(st, "g:0")["opacity"],
            "seg_opacity_g0_speaker90": seg_opacity(st, "0", "speaker90"),
            "seg_opacity_g0_speaker91": seg_opacity(st, "0", "speaker91"),
            "seg_opacity_g1": seg_opacity(st, "1"),
        }
    if "group_dim_again" in d and "after_dblclick" in d:
        out["dblclick_left_dim_unchanged"] = (out["group_dim_again"]["reference_label_opacity"] == out["after_dblclick"]["reference_label_opacity"] == "0.25"
                                              and out["after_dblclick"]["speaker90_label_opacity"] == "")
    ok = (out["speaker_dim"]["speaker90_label_opacity"] == "0.25" and out["speaker_dim"]["seg_opacity_g0_speaker90"] == ["0.25"]
          and out["speaker_dim"]["seg_opacity_g0_speaker91"] == [""] and out["speaker_dim"]["seg_opacity_g1"] == [""]
          and out["speaker_restored"]["seg_opacity_g0_speaker90"] == [""] and out["speaker_restored"]["speaker90_label_opacity"] == ""
          and out["group_dim"]["seg_opacity_g0_speaker90"] == ["0.25"] and out["group_dim"]["seg_opacity_g0_speaker91"] == ["0.25"]
          and out["group_dim"]["seg_opacity_g1"] == [""] and out["group_restored"]["seg_opacity_g0_speaker91"] == [""])
    out["ok"] = ok
    return out


def playback_summary(play, pause, replay, p_state, sp, space, back, after_back) -> dict:
    ro = [s[1] for s in play]
    ct = [s[2] for s in play]
    incs = [round(b - a, 3) for a, b in zip(ro, ro[1:])]
    ct_incs = [round(b - a, 3) for a, b in zip(ct, ct[1:])]
    pr = [s[1] for s in pause]
    pc = [s[2] for s in pause]
    rr = [s[1] for s in replay]
    sr_ = [s[1] for s in space]
    ab = [s[1] for s in after_back]
    return {
        "play_samples": len(play),
        "play_readout_first_last": [ro[0], ro[-1]],
        "play_readout_monotonic": all(i >= 0 for i in incs),
        "play_readout_max_increment_s": max(incs) if incs else None,
        "play_model_monotonic": all(i >= 0 for i in ct_incs),
        "play_model_max_increment_s": max(ct_incs) if ct_incs else None,
        "pause_playing_flag": p_state["playing"],
        "pause_readout_values": sorted(set(pr)),
        "pause_model_values": sorted(set(round(v, 6) for v in pc)),
        "pause_holds": len(set(pr)) == 1 and len(set(pc)) == 1,
        "replay_advance_over_1s": round(rr[-1] - rr[0], 3),
        "replay_elapsed_ms": replay[-1][0] - replay[0][0],
        "space_readout_before_sync_nextframe_100ms": [sp["before"], sp["sync"], sp["next_frame"], sp["after_100ms"]],
        "space_paused_within_one_frame": sp["next_frame"] == sp["after_100ms"] and not sp["playing_after_100ms"],
        "space_readout_values_next_300ms": sorted(set(sr_)),
        "space_pause_holds": len(set(sr_)) == 1,
        "back_while_playing": back,
        "back_while_playing_delta_raw_s": back["delta_raw"],
        "back_while_playing_delta_calibrated_s": back["delta_calibrated"],
        "back_then_keeps_increasing": all(b >= a for a, b in zip(ab, ab[1:])) and ab[-1] > ab[0] and back["later_300ms"] > back["stepped"],
        "back_then_first_last": [ab[0], ab[-1]],
    }


def window_duration(state: dict) -> float:
    return float(json.load(open(ROOT / "tests" / "fixture.json"))["duration"])


if __name__ == "__main__":
    main()
