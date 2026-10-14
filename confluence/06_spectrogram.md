<div dir="rtl">

<!-- cSpell:disable -->

# פרק 6 — ספקטרוגרמה

הכפתור `FFT` (או <span dir="ltr">`view="fft"`</span>) מחליף כל שורת אודיו בספקטרוגרמת mel שחושבה ב-Python. הזום מותח תמונה קיימת ולכן מיידי.

- STFT ב-numpy: חלון Hann באורך 512, קפיצה 128, בבלוקים של מיליון דגימות. כל דגימה נצפית; בהקטנה לרוחב המסך נשמר המקסימום, כך שפולסים קצרים נשארים.
- טווח דינמי של 80 dB מתחת ל-<span dir="ltr">`max(peak, −60 dBFS)`</span>; שקט נשאר כהה.
- ציר התדר: <span dir="ltr">`fscale="mel"`</span> (ברירת מחדל; התדרים הנמוכים מקבלים את רוב הגובה) או <span dir="ltr">`"linear"`</span>. ‏<span dir="ltr">`fmax=`</span> קובע את קצה הציר; 5000 Hz מתאים לדיבור.
- <span dir="ltr">`colormap=`</span> כל שם של matplotlib; ברירת מחדל `magma`.
- התמונה מדוגמת מחדש לגובה <span dir="ltr">`height`</span>. עמודה `i` מרוכזת ב-<span dir="ltr">`(i + 0.5) / cols`</span>, כמו הפונקציות.

## דוגמאות

<div dir="ltr">

```python
wavlens.show("audio.wav", "audio.rttm", view="fft", fmax=5000, height=160)
wavlens.show("audio.wav", "audio.rttm", view="fft", fscale="linear", colormap="viridis")
```

</div>

## מקרי שימוש

- **דיאריזציה.** הרמוניות (pitch) של דובר גבר נמוכות משל דוברת; החלפת דובר שקטה נראית בספקטרוגרמה גם כשהגל כמעט זהה.
- **שיפור דיבור.** השוואת ספקטרוגרמות של קלט ופלט מגלה "מוזיקליות" (רעש טונלי שנותר) שאינה נראית בגל.
- **הפרדה.** דליפה בין stems נראית כקווי הרמוניות של הדובר השני.
- **קודקים.** חיתוך תדרים גבוהים (<span dir="ltr">`fmax`</span> של המקור מול השחזור).

**צילום מסך להדבקה:** <span dir="ltr">`docs/spectrogram.png`</span> — ספקטרוגרמת mel של שני אודיו.

**צילום מסך להדבקה:** <span dir="ltr">`docs/zoom.png`</span> — זום של פי 40 עם ציר זמן צפוף.

</div>
