<div dir="rtl">

<!-- cSpell:disable -->

# פרק 9 — צינור דיאריזציה מלא

הרצת pipeline של pyannote.audio, השוואה לייחוס, האזנה לשגיאות.

<div dir="ltr">

```python
from pyannote.audio import Pipeline
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
import wavlens

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
hyp = pipeline("meeting.wav")
ref = load_rttm("meeting.rttm")["meeting"]
print(DiarizationErrorRate()(ref, hyp))

w = wavlens.show("meeting.wav", annotation={"reference": ref, "hypothesis": hyp},
                 function={"vad": vad_scores}, view="fft", fmax=5000, expanded=True)
w.solo_("reference/SPEAKER_01", "!hypothesis/SPEAKER_01")   # איפה המודל פספס את SPEAKER_01
```

</div>

השורה האחרונה משאירה בנגינה רק את הרגעים שבהם הייחוס אומר SPEAKER_01 וההיפותזה לא: שגיאות Missed Speech של אותו דובר.

## סדר עבודה מומלץ

1. <span dir="ltr">`show(file, {ref, hyp})`</span> ומבט כללי: היכן הצבעים אינם מתלכדים.
2. פתיחת החץ של ההיפותזה, לחיצה על דובר חשוד: האזנה לכל הקטעים שלו ברצף.
3. לחיצה נוספת (השלמה) על אותו דובר בייחוס: האזנה למה שהמודל ייחס לו בטעות.
4. `FFT` על קטע קצר (זום בגלגלת): בדיקה אם ההחלפה נראית בהרמוניות.
5. <span dir="ltr">`log_comet`</span> של המצב הנוכחי עם השלב של האימון.

## וריאציות

- **NeMo Sortformer.** <span dir="ltr">`annotation=model.diarize([file])[0]`</span> (שורות טקסט) או המטריצה `(T, K)` עם `threshold`.
- **WavLM.** <span dir="ltr">`annotation=model(**inputs)`</span>: ה-<span dir="ltr">`.logits`</span> עוברים סיגמואיד ובינאריזציה.
- **WhisperX.** רשומות המילים כקבוצה שלישית: מילים שנפלו על גבול דוברים.

</div>
