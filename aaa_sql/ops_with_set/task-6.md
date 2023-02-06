-- d7_set3 (n int);  
-- d7_set4 (n int);  
-- numbers (n int);  

Реализуйте аналог d7_set3 INTERSECT ALL d7_set4 через другие операции

UNION использовать нельзя.

Подсказка: для размножения строчек можно воспользоваться джойном на таблицу numbers (в ней числа от 1 до 50)

В постгрессе можно использовать конструкцию:
  SELECT *, generate_series(1, N) FROM …

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
select
    coalesce(d7_set3.n, d7_set4.n) as n
from numbers
    left join d7_set3 on 
        numbers.n = d7_set3.n and
        exists (
            select *
            from table_to_choose ttc
            where 
                ttc.is_first_tbl_argmin = 1 and
                numbers.n = ttc.n
        )
    left join d7_set4 on 
        numbers.n = d7_set4.n and
        exists (
            select *
            from table_to_choose ttc
            where 
                ttc.is_first_tbl_argmin = 0 and
                numbers.n = ttc.n
        )
where 
    d7_set3.n is not null or
    d7_set4.n is not null
```