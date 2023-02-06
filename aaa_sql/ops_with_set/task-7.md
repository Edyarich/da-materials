-- d7_set3 (n int);  
-- d7_set4 (n int);  
-- numbers (n int);  

Реализуйте аналог d7_set3 EXCEPT ALL d7_set4 через другие операции (UNION ALL можно использовать).

Подсказка: для размножения строчек можно воспользоваться джойном на таблицу numbers (в ней числа от 1 до 100)

В постгрессе можно использовать конструкцию:
  SELECT *, generate_series(1, N) FROM …

```
+---+
| n |
+---+
| 4 |
| 8 |
| 8 |
| 9 |
+---+
```

```sql
with count_table as (
    select
        counter3.n, 
        greatest(counter3.cnt - coalesce(counter4.cnt, 0), 0) as cnt
    from (
        select
            n, 
            count(n) as cnt
        from d7_set3
        group by n
    ) as counter3
    left join (
        select
            n,
            count(n) as cnt
        from d7_set4
        group by n
    ) as counter4 on
        counter3.n = counter4.n
)
select count_table.n
from numbers
join count_table on count_table.cnt >= numbers.n
order by 1
```