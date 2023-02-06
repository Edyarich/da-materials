-- d8_tournament_log (match_date, player_id, opponent_id, match_result, score);

Выведите игроков занявших призовые 1, 2 и 3 место по количеству побед.

Несколько игроков не могут занимать одно место: побеждает тот у кого суммарный score больше, учитывая score даже за проигранную партию

```
Ожидаемый формат ответа
+-------+-----------+--------------+-------------+
| place | player_id | total_result | total_score |
+-------+-----------+--------------+-------------+
| 1     | 2         | 4            | 249         |
| 2     | 1         | 2            | 266         |
| 3     | 3         | 2            | 249         |
+-------+-----------+--------------+-------------+
```

```sql
with scores_info as (
    select
        player_id,
        sum(case match_result
            when 1 then 1
            else 0
        end) as total_result,
        sum(score) as total_score
    from d8_tournament_log
    group by player_id
), 
places as (
    select
        dense_rank() over (
            order by 
                total_result desc,
                total_score desc
        ) as place,
        player_id,
        total_result,
        total_score
    from scores_info
)
select *
from places
where place <= 3
order by 
    total_result desc, 
    total_score desc
```