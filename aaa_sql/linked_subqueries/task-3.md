Дана табличка с транзакциями пользователей.

```sql
CREATE TABLE transactions_day6 (
	id int,
	user_id int4,
	amount int4,
	dtime timestamp
);
```

Нужно найти наибольшую сумму транзакций, которые сделал юзер за 10 суток (max_sum_10_day).
Решите задачу через связанный подзапрос в select-списке.

Можно считать, что у одного юзера нет 2 транзакций на один timestamp (timestamp = дата+время).
Для работы с интервалами в формате timestamp - воспользуйтесь командой interval. Например, чтобы получить предыдущий день, надо взять dtime - interval 1 day.

```
Формат ответа 
| user_id | max_sum_10_day |
| ------- | -------------- |
| 1       | 10123          |
| 2       | 90218          |
```

**Ограничения**: запрещено использовать LIMIT, UNION, UNION ALL, VALUES, IN, JOIN

```sql
SELECT 
    user_id,
    MAX(sum_10_day) AS max_sum_10_day
FROM (
    SELECT 
        td.user_id,
        td.dtime,
        (
            SELECT SUM(td2.amount)
            FROM transactions_day6 td2
            WHERE
                td.user_id = td2.user_id AND
                td2.dtime >= td.dtime AND 
                td2.dtime < td.dtime + INTERVAL 10 DAY
        ) AS sum_10_day
    FROM transactions_day6 td 
) AS sum_10_day_table
GROUP BY user_id
```