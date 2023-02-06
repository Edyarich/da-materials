-- d8_scores (event_date, category, subcategory, value)

Для дат с 2021-01-19 по 2021-01-21 выведите среднее значение value за предыдущие два дня для каждой подкатегории категории A

Например, для даты 2021-01-19 вычисление среднего должно происходить по датам 2021-01-17, 2021-01-18.

```
Ожидаемый формат ответа:
+------------+----------+-------------+-------+-------------+
| event_date | category | subcategory | value | last_2d_avg |
+------------+----------+-------------+-------+-------------+
| 2021-01-19 | A        | A:X         | 555   | 164.0000    |
| 2021-01-19 | A        | A:Y         | 912   | 234.0000    |
| 2021-01-20 | A        | A:X         | 828   | 410.5000    |
| 2021-01-20 | A        | A:Y         | 378   | 688.5000    |
| 2021-01-21 | A        | A:X         | 369   | 691.5000    |
| 2021-01-21 | A        | A:Y         | 885   | 645.0000    |
+------------+----------+-------------+-------+-------------+
```

```sql
select *
from (
    select
        event_date,
        category,
        subcategory,
        value,
        avg(value) over (
            partition by subcategory
            order by event_date
            range between interval 2 day preceding and interval 1 day preceding
        ) as last_2d_avg
    from d8_scores
    where category = 'A'
) all_days_result
where event_date between '2021-01-19' and '2021-01-21'
order by event_date, category, subcategory
```