-- d8_scores (event_date, category, subcategory, value)

Посчитайте какую долю составляет суммарное значение по подкатегории от значения по категории в процентах

```
Ожидаемый формат ответа
+----------+-------------+-------------------+-----------------+
| category | subcategory | subcategory_value | subcategory_pct |
+----------+-------------+-------------------+-----------------+
| A        | A:X         | 9851              | 50.07880        |
| A        | A:Y         | 9820              | 49.92120        |
| B        | B:X         | 11367             | 49.88371        |
| B        | B:Y         | 11420             | 50.11629        |
+----------+-------------+-------------------+-----------------+
```

```sql
select distinct
    category,
    subcategory,
    sum(value) over (
        partition by subcategory
    ) as subcategory_value,
    sum(value) over (
        partition by subcategory
    ) / sum(value) over (
        partition by category
    ) * 100.0 as subcategory_pct
from d8_scores
order by 1, 2
```
