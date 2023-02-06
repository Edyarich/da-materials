Даны 3 таблицы. Юзеры, Локации, Клики.  
Нужно найти пользователей, у которых помимо своей есть еще локация, на объявления которой, они кликают чаще.

```sql
create table d10_hw_users
(
    id          int,
    name        varchar(64),
    location_id int
);

create table d10_hw_locations
(
    id          int,
    city        varchar(64)
);

create table d10_hw_clicks
(
    log_id      int,
    user_id     int,
    location_id int,
    category_id int
);
```

Требуется выполнить следующие действия

- Добавить колонку `second_location_id int` в таблицу `d10_hw_users`
- Создать таблицу `d10_hw_temp`, в которую записать пользователя и локацию, на которую он кликал чаще чем на другие локации. Структура новой таблицы `(user_id, location_id)`. У каждого пользователя должна быть одна такая локация, если количество кликов одинаковое - выберете локацию с наименьшей id.
- Заполните `d10_hw_users.second_location_id` в соответствии с предыдущей таблицей
- Удалите тех пользователей, у которых вторая локация совпадает с оригинальной
- В последнем запросе выведите топ 3 категории, которые просматривали эти оставшиеся пользователи в порядке убывания общего числа просмотров

```
Query result:
+-------------+
| category_id |
+-------------+
| 1           |
| 0           |
| 3           |
+-------------+
```

```sql
alter table d10_hw_users add column second_location_id int;

create table d10_hw_temp
as
with count_table as (
    select distinct
        user_id,
        location_id,
        count(log_id) over (
            partition by user_id, location_id
        ) as location_counter
    from d10_hw_clicks
)
select
    user_id,
    min(location_id) as location_id
from count_table ct1
where location_counter >= all (
    select location_counter
    from count_table ct2
    where ct1.user_id = ct2.user_id
)
group by user_id;

update d10_hw_users u
set second_location_id = (
    select location_id
    from d10_hw_temp tmp
    where u.id = tmp.user_id
)
where true;

delete from d10_hw_users
where location_id = second_location_id;

select category_id
from d10_hw_clicks
where user_id in (
    select id
    from d10_hw_users
)
group by category_id
order by count(log_id) desc
limit 3;
```
