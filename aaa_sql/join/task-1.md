В таблице trips содержатся поездки таксопарка ООО 'КЕХ Ромашка'. Для каждой поездки указаны client_id и driver_id, которые являются внешними ключами на таблицу users (поле user_id). В поле status могут содержаться значения ('completed', 'cancelled_by_driver', 'cancelled_by_client').

Таблица users содержит всех пользователей таксопарка (и клиентов и водителей). В поле role указана их роль в таксопарке, а в поле banned их статус блокировки.

Напишите запрос, который найдет коэффициент отмены в промежутке между 2020-02-01 и 2020-02-03. Коэффициент отмены - отношение отмененных поездок к общему количеству поездок. Учитывать нужно только незаблокированных пользователей (незаблокированы должны быть и клиент и водитель).

Решите задачу используя только один JOIN (любого типа), подзапросы с IN использовать нельзя, подзапрос для группировки - можно.

```
Ожидаемый формат ответа:
 request_at |      cancel_rate       
------------+------------------------
 2020-02-01 | 0.33
 2020-02-02 | 0.00
 2020-02-03 | 0.50
```

cancel_rate округлите до двух знаков после запятой, при этом число должно получиться меньше или равное исходному (воспользуйтесь функцией TRUNCATE)

**Ограничения**: запрещено использовать, LIMIT, UNION, UNION ALL, VALUES, IN.

```sql
SELECT
    request_at,
    truncate(
        avg(
            CASE status
                WHEN 'completed' THEN 0
                ELSE 1
            END
        ), 2
    ) AS cancel_rate
FROM (
    SELECT 
        driver_id,
        client_id,
        status, 
        request_at
    FROM trips t 
        JOIN users u ON 
            (u.user_id = t.client_id 
            OR u.user_id = t.driver_id)
            AND u.banned = 0
    WHERE 
        request_at BETWEEN '2020-02-01' AND '2020-02-03'
    GROUP BY
        driver_id,
        client_id,
        status, 
        request_at
    HAVING count(*) = 2
) unbanned_trips
GROUP By request_at
```