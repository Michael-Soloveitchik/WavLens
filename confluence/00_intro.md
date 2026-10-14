<div dir="rtl">

<!-- cSpell:disable -->

# WavLens — הצגה אינטראקטיבית של אודיו ודיאריזציה ב-Jupyter

**WavLens** היא ספריית Python קטנה. קריאה אחת, <span dir="ltr">`wavlens.show(...)`</span>, מציירת במחברת גל קול או ספקטרוגרמה, ציר זמן, שורה לכל דובר, עקומות של פונקציות וקו נגינה אדום נע. הקלט הוא האובייקטים של pyannote, ‏NeMo ו-WavLM כפי שהם, או קובצי RTTM / UEM / LAB / SegLST / JSON.

קוד פתוח: [github.com/Michael-Soloveitchik/WavLens](https://github.com/Michael-Soloveitchik/WavLens). רישיון MIT.

## התקנה

<div dir="ltr">

```bash
pip install wavlens
pip install git+https://github.com/Michael-Soloveitchik/WavLens.git   # גרסה אחרונה
```

</div>

Python 3.10 ומעלה. כל חזית Jupyter. התלויות (anywidget, numpy, soundfile, pyannote.core, matplotlib, scipy) מותקנות אוטומטית.

## שלוש שורות

<div dir="ltr">

```python
import wavlens
wavlens.show("audio.wav", "audio.rttm")
```

</div>

**צילום מסך להדבקה:** <span dir="ltr">`docs/overview.png`</span> — שני אודיו, שורת mix, ייחוס והיפותזה, עקומת RMS.

<div dir="ltr">

```python
wavlens.show(audio={"clean": (x, sr), "noisy": (y, sr)},
             annotation={"reference": "audio.rttm", "hypothesis": pipeline(file)},
             function={"rms": rms}, expanded=["reference"])
```

</div>

## מה רואים ומה עושים

| פעולה | תוצאה |
|---|---|
| גלגלת העכבר | זום סביב הסמן |
| לחיצה על הגל | קפיצה לנקודה; מצב הנגינה לא משתנה |
| רווח, ← / → | נגינה/השהיה, הזזה ב-10 מילישניות |
| `FFT` | ספקטרוגרמת mel במקום הגל |
| תיבת סימון של אודיו | השתקה; השורה מולבנת |
| לחיצה על שם דובר או אנוטציה | השארת הקטעים שלו → השלמה → רגיל. כמה שמות = חיתוך |

**צילום מסך להדבקה:** <span dir="ltr">`docs/solo.png`</span> — חיתוך של speaker91 פחות speaker90; האודיו noisy מושתק.

## למה זה חשוב

מדד כמו DER מסכם ניסוי במספר אחד. המספר אינו אומר היכן המודל טעה ומה נשמע שם. WavLens מאפשרת להאזין בדיוק לקטעים שבהם הייחוס וההיפותזה חלוקים, להשוות כמה מודלים על אותו ציר זמן, ולצרף אותות עזר כמו VAD או הסתברות דובר. ההמחשה נשמרת כ-HTML עצמאי ונרשמת ל-Comet או ל-Weights & Biases בשורה אחת.

## הפרקים

| קובץ | נושא |
|---|---|
| 01_audio_inputs | צורות קלט של אודיו |
| 02_annotation_inputs | צורות קלט של אנוטציות |
| 03_functions | פונקציות (עקומות) |
| 04_functions_per_audio | פונקציות הקשורות לאודיו מסוים |
| 05_layout | פריסה, מסכות, גרירה, שורת mix |
| 06_spectrogram | ספקטרוגרמה |
| 07_weights_interpolation | משקלים ואינטרפולציה |
| 08_python_control | שליטה מ-Python וייצוא |
| 09_real_pipeline | צינור דיאריזציה מלא |
| 10_use_cases | מקרי שימוש לפי משימה |
| 11_glossary | מילון מונחים |

תיעוד מלא באנגלית במאגר: <span dir="ltr">`FUNCTIONALITIES.md`</span>, ‏<span dir="ltr">`FIXED_BUGS.md`</span>, ‏<span dir="ltr">`examples/wavlens.ipynb`</span>.

</div>
