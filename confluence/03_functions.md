<div dir="rtl">

<!-- cSpell:disable -->

# פרק 3 — פונקציות

<span dir="ltr">`function=`</span> מצייר עקומות מעל שורות האודיו. הקלט: אות חד-ממדי, מטריצת פריימים `(T, K)` (עקומה לכל עמודה), ‏<span dir="ltr">`.logits`</span>, ‏`SlidingWindowFeature` של pyannote, רשימה או מילון <span dir="ltr">`{name: item}`</span>.

- כל עקומה נמתחת לטווח 0–1 (min-max). אות קבוע מצויר ב-0.5.
- כל אורך מתקבל. בהקטנה ל-2000 נקודות נשמר הערך המוחלט הגדול ביותר בכל חלון, כך שפולסים קצרים שורדים. בהגדלה נעשית אינטרפולציה לפי <span dir="ltr">`interpolation`</span>.
- מוסכמת הזמן: דגימה `k` מתוך `L` יושבת ב-<span dir="ltr">`(k + 0.5) / L`</span> מהמשך. פריים של 20 מילישניות מצויר במרכזו.
- השם מופיע כ-tooltip. הצבעים מחזוריים: שחור, כחול, אדום, ירוק, סגול, חום.

## דוגמאות

<div dir="ltr">

```python
wavlens.show("audio.wav", "audio.rttm", function={"vad": vad_prob, "energy": rms, "f0": pitch})
wavlens.show("audio.wav", function=segmentation_model(file))        # SlidingWindowFeature: עקומה לכל דובר
wavlens.show("audio.wav", function=wavlm_out.logits)                 # (B, T, K): עקומה לכל עמודה
```

</div>

## מקרי שימוש

- **SAD / VAD.** הסתברות דיבור של המודל מול הקטעים של הייחוס; ספים שגויים נראים כעקומה שחוצה 0.5 בתוך קטע.
- **אימות דוברים (Speaker Verification).** ציון דמיון של חלון נע מול embedding של דובר יעד; עליות של הציון צריכות להתלכד עם הקטעים של אותו דובר.
- **סגמנטציה של pyannote.** פלט <span dir="ltr">`SlidingWindowFeature`</span> של מודל הסגמנטציה: עקומה לכל דובר מקומי, ישירות ללא המרה.
- **איכות אות.** SNR רגעי, אנרגיה, ‏pitch; מסבירים למה דובר שקט פוספס.

**צילום מסך להדבקה:** <span dir="ltr">`docs/functions.png`</span> — שלוש עקומות (VAD, אנרגיה, f0) על אודיו אחד.

</div>
