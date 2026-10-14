"""Build tests/fixture.json (+ variants) from examples/data/sample.wav + sample.rttm.

Runs the real ``wavlens.show()`` and dumps every synced trait, so the fixture is
exactly what the browser side would receive from anywidget.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from pyannote.database.util import load_rttm  # noqa: E402

import wavlens  # noqa: E402

DATA = ROOT / "examples" / "data"
HERE = Path(__file__).resolve().parent


def dump(w, out: Path) -> None:
    # anywidget subclasses per instance, so ask the declared class; skips ipywidgets' layout etc.
    names = sorted(wavlens.WavLens.class_own_traits(sync=True))
    fixture = {n: getattr(w, n) for n in names}
    out.write_text(json.dumps(fixture))
    nseg = sum(len(g["segments"]) for g in fixture["groups"])
    rows = [(g.get("name"), g["rows"]) for g in fixture["groups"]]
    fns = [(f["name"], f.get("row")) for f in fixture.get("functions", [])]
    sizes = {n: len(v) for n, v in fixture.items() if isinstance(v, (list, str))}
    print(f"wrote {out} traits={names}\n  duration={fixture['duration']:.2f}s "
          f"channels={fixture.get('channels')} segments={nseg} rows={rows} functions={fns} sizes={sizes}")


def main() -> None:
    ann = load_rttm(DATA / "sample.rttm")["sample"]
    x, sr = sf.read(DATA / "sample.wav", dtype="float32")
    rms = np.sqrt(np.convolve(x**2, np.ones(400) / 400, "same"))
    # Default fixture: two audios, two annotation groups, one function on every row.
    dump(wavlens.show(audio={"clean": (x, sr), "quiet": (x * 0.3, sr)},
                    annotation={"reference": ann, "speech": ann.get_timeline()},
                    function={"rms": rms}),
         HERE / "fixture.json")
    # Function-row variant: "clean" binds to audio row 0, "glob" is drawn on every row and lane group.
    dump(wavlens.show(audio={"clean": (x, sr), "quiet": (x * 0.3, sr)}, annotation=ann,
                    function={"clean": rms, "glob": rms[::100]}),
         HERE / "fixture_fn_rows.json")
    # Three-group variant for drag-to-reorder: a (two speakers), b (blank-label timeline), c (two speakers again).
    dump(wavlens.show(audio={"clean": (x, sr)}, annotation={"a": ann, "b": ann.get_timeline(), "c": ann}),
         HERE / "fixture_3groups.json")


if __name__ == "__main__":
    main()
