-- d8_player_city (player_id, city, actual_date)

Найдите игроков, возвращавшихся в свой первый город.

Игрок повторно отмечающийся в том же городе не считается вернувшимся

```
Ожидаемый формат ответа:
+-----------+
| player_id |
+-----------+
| 5         |
| 6         |
+-----------+
```

```sql
with city_pairs as (
    select
        player_id,
        city,
        first_value(city) over (
            partition by player_id
            order by actual_date
        ) as first_city,
        lead(city) over (
            partition by player_id
            order by actual_date
        ) as next_city
    from d8_player_city
    order by
        player_id, 
        actual_date
)
select distinct player_id
from city_pairs
where
    next_city = first_city and
    city != next_city
```