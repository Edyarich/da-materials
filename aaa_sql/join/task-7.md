Дана таблички с транзакциями пользователей.

```sql
CREATE TABLE transactions (
	id int,
	user_id int4,
	amount int4,
	dtime timestamp
);
```
Нужно найти наибольшее число транзакций, которые сделал юзер за 30 суток (max_count_30_day).

Можно считать, что у одного юзера нет 2 транзакция на один timestamp (timestamp = дата+время).
Для работы с интервалами в формате timestamp - воспользуйтесь командой interval. Например, чтобы получить предыдущий день, надо взять dtime - interval 1 day.

```
Формат ответа 
| user_id | max_count_30_day |
| ------- | ---------------- |
| 1       | 10               |
| 2       | 18               |
```

**Ограничения**: запрещено использовать связанные подзапросы (обращение в подзапросе к таблицам из внешнего запроса), LIMIT, UNION, UNION ALL, VALUES, IN

```sql
SELECT 
    user_id, 
    MAX(count_30_day) as max_count_30_day
FROM (
    SELECT 
        t.user_id,
        t.dtime,
        count(1) as count_30_day
    FROM transactions t 
    JOIN transactions prev_t 
        ON
            t.dtime >= prev_t.dtime
            AND t.dtime <= prev_t.dtime + INTERVAL 30 DAY
            AND t.user_id = prev_t.user_id
    GROUP BY 
        t.user_id, 
        t.dtime
) count_30_day_tbl
GROUP BY user_id
```