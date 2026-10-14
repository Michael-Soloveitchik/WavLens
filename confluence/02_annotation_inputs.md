<div dir="rtl">

<!-- cSpell:disable -->

# פרק 2 — צורות קלט של אנוטציות

כל פריט הופך לקבוצת נתיבים אחת: שורה לכל תווית, לפי הסדר של <span dir="ltr">`Annotation.labels()`</span>. הקטעים נחתכים לטווח האודיו.

| צורה | זיהוי |
|---|---|
| <span dir="ltr">`pyannote.core.Annotation`</span>, ‏`Timeline`, ‏`Segment` | לפי טיפוס |
| `DiarizeOutput` ‏(pyannote.audio 4) | התכונה <span dir="ltr">`.speaker_diarization`</span> |
| <span dir="ltr">`{"diarization": [{start, end, speaker}, ...]}`</span> | פלט <span dir="ltr">`serialize()`</span> |
| <span dir="ltr">`{"rttm_filepath": ...}`</span> | רשומת manifest של NeMo; ה-RTTM נקרא |
| <span dir="ltr">`["0.00 4.98 speaker_0", ...]`</span> | שורות NeMo Sortformer; רשימה של רשימות = קבוצה לכל קובץ |
| <span dir="ltr">`[{"start", "end", "speaker"}, ...]`</span> | רשומות SegLST או WhisperX |
| <span dir="ltr">`[(start, end, label), ...]`</span> או <span dir="ltr">`[(start, end), ...]`</span> | רשימת tuples |
| <span dir="ltr">`{speaker: [(start, end), ...]}`</span> | אנוטציה **אחת**; המפתחות הם הדוברים |
| טקסט RTTM | מחרוזת שמתחילה ב-<span dir="ltr">`SPEAKER `</span> |
| <span dir="ltr">`.logits`</span> בצורה `(B, T, K)` | פלט WavLM; סיגמואיד ובינאריזציה ב-`threshold` |
| מטריצת פריימים `(T, K)` | רק דרך <span dir="ltr">`annotation=`</span>; תוויות <span dir="ltr">`spk<k>`</span> |

קבצים לפי סיומת: `.rttm`, ‏`.uem`, ‏`.lab` / `.txt`, ‏`.json` / `.seglst`, ‏`.jsonl`.

**כלל ההבחנה:** מילון שערכיו נושאים תוויות משלהם (tuples של שלושה, רשומות, אובייקטים, נתיבים) הוא מילון של אנוטציות בשם. מילון שערכיו זוגות `(start, end)` בלבד הוא אנוטציה אחת והמפתחות הם הדוברים.

## דוגמאות

<div dir="ltr">

```python
wavlens.show("audio.wav", annotation={"reference": "audio.rttm", "hypothesis": pipeline(file)})
wavlens.show("audio.wav", annotation={"A": [(0.5, 2.0), (4.0, 5.5)], "B": [(2.0, 4.0)]})   # אנוטציה אחת
wavlens.show("audio.wav", annotation=model(**inputs))                                   # WavLM logits
wavlens.show("audio.wav", annotation=sortformer_probs, threshold=0.6)                   # (T, K)
```

</div>

## מקרי שימוש

- **דיאריזציה.** ייחוס מול היפותזה, שתי קבוצות באותו ציר. שגיאות החלפת דוברים נראות כפסים בצבע שונה באותו זמן.
- **זיהוי דוברים (Speaker Recognition).** תוצאת הזיהוי כאנוטציה אחת עם שם הדובר שזוהה בכל קטע; פס אפור לקטעים שסווגו כ"לא ידוע".
- **זיהוי פעילות קולית (SAD / VAD).** `Timeline` של דיבור או מטריצת `(T, 1)` של הסתברויות עם `threshold`.
- **יישור ASR.** רשומות WhisperX לפי מילה או משפט כקבוצה נוספת מתחת לדוברים.
- **אזורי ניקוד.** קובץ `.uem` כשורה אחת: היכן המדד מחושב.

**צילום מסך להדבקה:** <span dir="ltr">`docs/compare.png`</span> — ייחוס מול היפותזה, צבעים לפי אנוטציה.

**צילום מסך להדבקה:** <span dir="ltr">`docs/many.png`</span> — חמש אנוטציות, שמונה דוברים בכל אחת.

</div>
