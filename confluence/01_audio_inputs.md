<div dir="rtl">

<!-- cSpell:disable -->

# פרק 1 — צורות קלט של אודיו

כל אודיו נושא את קצב הדגימה שלו. מערך חשוף ללא קצב מעלה שגיאה. זו החלטת תכן: קצב שגוי מזיז את כל הקטעים בשקט.

| צורה | קצב הדגימה | הערות |
|---|---|---|
| <span dir="ltr">`"path.wav"`</span> או `Path` | מהקובץ | כל פורמט של libsndfile; ערוצים מרובים ממוצעים למונו |
| <span dir="ltr">`{"audio": path}`</span> | מהקובץ | ‏AudioFile של pyannote |
| <span dir="ltr">`{"waveform": tensor, "sample_rate": sr}`</span> | `sr` | טנזור `(C, T)`; גם על GPU |
| <span dir="ltr">`(array_or_tensor, sr)`</span> | `sr` | מה ש-<span dir="ltr">`torchaudio.load()`</span> מחזיר |
| <span dir="ltr">`(path, sr)`</span> | מהקובץ, מדוגם מחדש ל-`sr` | |
| רשימה או מילון של הצורות לעיל | לכל פריט | מפתחות המילון הם שמות השורות |

כללים:

- טנזורים עוברים <span dir="ltr">`.detach().cpu()`</span>. שום דבר לא מועבר ל-GPU.
- שלמים מחולקים במקסימום שלהם (int16 → ±1). ‏NaN או inf מעלים שגיאה עם הספירה.
- <span dir="ltr">`sample_rate=`</span> הוא יעד אופציונלי. בלעדיו, קצבים שונים מדוגמים מחדש לקצב הגבוה ביותר, כך שלא אובד מידע.
- כל האודיו חייב להיות באותו אורך. הפרש של עד <span dir="ltr">`max(2, sr/1000)`</span> דגימות נחתך; יותר מזה מעלה שגיאה.
- שמות ברירת מחדל: `Audio`, ‏<span dir="ltr">`Audio_<i>`</span>. התנגשות שמות מקבלת סיומת `~1`.

## דוגמאות

<div dir="ltr">

```python
import soundfile as sf, torch
x, sr = sf.read("audio.wav", dtype="float32")

wavlens.show((x, sr), "audio.rttm")                                        # tuple
wavlens.show({"waveform": torch.from_numpy(x)[None], "sample_rate": sr})    # pyannote AudioFile
wavlens.show(audio={"clean": (x, sr), "noisy": (x + noise, sr)})          # שמות שורות
wavlens.show(["a.wav", "b.wav"], sample_rate=16000)                        # יעד משותף
```

</div>

## מקרי שימוש

- **הפרדת דוברים (Speaker Separation).** התערובת ושני ה-stems שהמודל הוציא כשלוש שורות: <span dir="ltr">`audio={"mix": (m, sr), "spk1": (s1, sr), "spk2": (s2, sr)}`</span>. תיבות הסימון מאפשרות לשמוע כל stem לבד או יחד.
- **שיפור דיבור (Enhancement).** קלט מורעש ופלט משופר זה מעל זה, עם האנוטציה של הייחוס, כדי לבדוק אם השיפור מחק דיבור שקט.
- **קידוד/דחיסה.** המקור והשחזור של codec; ההבדל נשמע בשורת mix אם המשקל של אחד מהם שלילי (ראו פרק 7).

**צילום מסך להדבקה:** <span dir="ltr">`docs/mix.png`</span> — שלושה אודיו (נקי, מורעש, שקט) ושורת mix.

</div>
