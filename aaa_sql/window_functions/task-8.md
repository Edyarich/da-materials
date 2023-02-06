-- create table d9_payments (dt timestamp, user_id int, amnt int, platform_id int, log_id int);  
-- create table d9_platform (id int, nm text);  

Дана таблица с платежами пользователей выполненных с различных платформ.

Для каждой платформы найдите пользователя с наибольшей средней транзакцией в рамках трехдневного окна.

Ответ округлите до целого с помощью функции `round`

Дополнительно выведите минимальный и максимальный идентификатор строки (log_id) в рамках окна.

Дополнительно выведите дату начала окна и дату конца

```
+---------------+---------+---------------+----------------+---------------+-----------------+----------------+
| platfrom_name | user_id | avg_amnt_w_3d | first_row_w_3d | last_row_w_3d | first_date_w_3d | last_date_w_3d |
+---------------+---------+---------------+----------------+---------------+-----------------+----------------+
| android       | 4       | 478           | 4003           | 4003          | 2021-01-08      | 2021-01-08     |
| ios           | 2       | 475           | 2004           | 2007          | 2021-01-03      | 2021-01-05     |
| web           | 6       | 778           | 6001           | 6002          | 2021-01-01      | 2021-01-03     |
+---------------+---------+---------------+----------------+---------------+-----------------+----------------+
```

Обратите внимание как именно считается конец кадра при указании range between ... and current row. **Необходимо чтобы все платежи одного дня попадали в один кадр**.

Обратите внимание на приведение типов: cast(column as date)

```sql
with total_info as (
    with payments as (
        select
            dt,
            user_id,
            amnt,
            nm as platfrom_name,
            log_id
        from d9_payments d_pay
        left join d9_platform d_plat on
            d_pay.platform_id = d_plat.id
    )
    select
        platfrom_name,
        user_id,
        round(
            avg(amnt) over (
                partition by
                    platfrom_name,
                    user_id
                order by cast(dt as date)
                range between interval 2 day preceding and current row
            )
        ) as avg_amnt_w_3d,
        min(log_id) over (
            partition by
                platfrom_name,
                user_id
            order by cast(dt as date)
            range between interval 2 day preceding and current row
        ) as first_row_w_3d,
        max(log_id) over (
            partition by
                platfrom_name,
                user_id
            order by cast(dt as date)
            range between interval 2 day preceding and current row
        ) as last_row_w_3d,
        first_value(cast(dt as date)) over (
            partition by
                platfrom_name,
                user_id
            order by cast(dt as date)
            range between interval 2 day preceding and current row
        ) as first_date_w_3d,
        last_value(cast(dt as date)) over (
            partition by
                platfrom_name,
                user_id
            order by cast(dt as date)
            range between interval 2 day preceding and current row
        ) as last_date_w_3d
    from payments
)
select distinct *
from total_info ti
where avg_amnt_w_3d >= all (
    select avg_amnt_w_3d
    from total_info ti2
    where ti.platfrom_name = ti2.platfrom_name
)
```
