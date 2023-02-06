-- d7_set3 (n int);  
-- d7_set4 (n int);  
-- numbers (n int);  

Реализуйте аналог d7_set3 INTERSECT ALL d7_set4 через другие операции (UNION ALL можно использовать).
Результат отсортируйте по возрастанию.

Подсказка: для размножения строчек можно воспользоваться джойном на таблицу numbers (в ней числа от 1 до 100)

В постгрессе можно использовать конструкцию:
  SELECT *, generate_series(1, N) FROM …
```
+----+
| n  |
+----+
| 5  |
| 6  |
| 7  |
| 9  |
| 10 |
| 10 |
| 11 |
| 11 |
| 11 |
| 12 |
| 12 |
| 12 |
| 12 |
| 12 |
| 13 |
| 13 |
| 13 |
| 13 |
| 13 |
+----+
```

```sql
with table_to_choose as (
    select
        counter3.n,
        case least(counter3.cnt, counter4.cnt)
            when counter3.cnt then 1
            else 0
        end as is_first_tbl_argmin
    from (
        select
            n, 
            count(n) as cnt
        from d7_set3
        group by n
    ) as counter3
    join (
        select
            n,
            count(n) as cnt
        from d7_set4
        group by n
    ) as counter4 on
        counter3.n = counter4.n
)
select *
from d7_set3 s3
where exists (
    select *
    from table_to_choose ttc
    where 
        s3.n = ttc.n and
        ttc.is_first_tbl_argmin = 1
)
union all
select *
from d7_set4 s4
where exists (
    select *
    from table_to_choose ttc
    where 
        s4.n = ttc.n and
        ttc.is_first_tbl_argmin = 0
)
order by n
```