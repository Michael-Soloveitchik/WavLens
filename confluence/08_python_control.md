<div dir="rtl">

<!-- cSpell:disable -->

# פרק 8 — שליטה מ-Python וייצוא

כל מצב של הממשק זמין כארגומנט של <span dir="ltr">`show()`</span> וכתכונה (Trait) של הווידג'ט.

| ארגומנט | משמעות |
|---|---|
| <span dir="ltr">`enabled=[True, False]`</span> | אילו אודיו דולק |
| <span dir="ltr">`solo=["ref", "!ref/speaker91"]`</span> | מסכות; `!` = השלמה |
| <span dir="ltr">`dim=["hyp/A"]`</span> | עמעום (רק מ-Python) |
| <span dir="ltr">`expanded=True`</span> או רשימת שמות | אילו אנוטציות פתוחות |
| <span dir="ltr">`order=[1, 0]`</span> | סדר שורות האודיו |
| <span dir="ltr">`zoom=40`</span> | פיקסלים לשנייה; 0 = התאמה לרוחב |
| <span dir="ltr">`time=10.2`</span> | מיקום התחלתי |
| <span dir="ltr">`autoplay=True`</span> | נגינה אוטומטית; דפדפנים עשויים לחסום עד לחיצה |
| <span dir="ltr">`view="fft"`</span> | תצוגה התחלתית |
| <span dir="ltr">`step=0.01`</span> | גודל הצעד של החיצים |

שיטות: <span dir="ltr">`w.seek(t)`</span>, ‏<span dir="ltr">`w.play()`</span>, ‏<span dir="ltr">`w.pause()`</span>, ‏<span dir="ltr">`w.solo_("ref", "ref/speaker90")`</span> (ללא ארגומנטים = ניקוי), ‏<span dir="ltr">`w.dim(...)`</span>, ‏<span dir="ltr">`w.enabled = [...]`</span>, ‏<span dir="ltr">`w.expanded = ["ref"]`</span>, ‏<span dir="ltr">`w.groups`</span> (הסדר הנוכחי). התכונות <span dir="ltr">`current_time`</span>, ‏`playing`, ‏`enabled`, ‏`solo`, ‏`expanded`, ‏`order` מסונכרנות בשני הכיוונים. השהיה כותבת את המיקום המדויק ל-Python.

<div dir="ltr">

```python
w = wavlens.show("audio.wav", "audio.rttm", solo="sample/speaker90", time=7.0)
w.seek(12.5); w.play()
w.current_time           # מתעדכן מהדפדפן
```

</div>

## ייצוא: HTML, ‏Comet, ‏Weights & Biases

- <span dir="ltr">`w.html()`</span> מחזיר קובץ HTML עצמאי אחד: הקוד, כל הנתונים ומודל בתוך הדף. אודיו, ספקטרוגרמות, מסכות, זום ומקלדת עובדים ללא קרנל. <span dir="ltr">`w.save_html("run.html")`</span> כותב לדיסק.
- <span dir="ltr">`w.log_comet(experiment, name="wavlens", step=epoch)`</span> רושם ללשונית HTML של הניסוי וכנכס.
- <span dir="ltr">`w.log_wandb(key="diarization", step=epoch)`</span> רושם <span dir="ltr">`wandb.Html`</span> שמוצג בדף הריצה.
- גודל: כ-1.3 MB לכל 30 שניות של מונו ב-16 kHz, ועוד 64 KB של קוד.

<div dir="ltr">

```python
for epoch in range(epochs):
    train(...)
    wavlens.show(file, {"hyp": model.diarize(file), "ref": ref}).log_comet(exp, step=epoch)
```

</div>

## מקרי שימוש

- **מעקב אימון.** אותו קובץ בדיקה בכל epoch ב-Comet: רואים ושומעים מתי המודל התחיל להפריד דוברים.
- **סקירה של תיוג.** <span dir="ltr">`save_html`</span> לכל קובץ שהמתייג צריך לבדוק; נפתח בדפדפן ללא Python.
- **לולאת ניפוי.** סקריפט שמוצא את ה-DER הגרוע ביותר ברשימת קבצים ופותח את הווידג'ט עם <span dir="ltr">`solo=`</span> על הדובר הבעייתי ו-<span dir="ltr">`time=`</span> על השגיאה הראשונה.

</div>
