Дана таблица slack с одним полем slack_name.
Напишите запрос, который выведет все неповторяющиеся логины (встречаются единожды, и только их) в любом порядке.

**Ограничения**: запрещено использовать HAVING, JOIN, LIMIT, UNION, UNION ALL, VALUES.

```sql
SELECT slack_name
FROM (
    SELECT 
        slack_name, 
        count(slack_name) AS cnt
    FROM slack
    GROUP BY slack_name
) AS names_count
WHERE 
    cnt = 1
```