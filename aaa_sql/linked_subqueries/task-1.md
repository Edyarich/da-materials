Дана таблица sequence (id, num).

Напишите запрос, который найдет все числа (вывести только уникальные), которые появляются в таблице последовательно (при сортировке по id) как минимум три раза подряд.

```
Ожидаемый формат ответа:
 num
----------------
40
5
```

**Ограничения**: запрещено использовать HAVING, связанные подзапросы (обращение в подзапросе к таблицам из внешнего запроса), LIMIT, UNION, UNION ALL, VALUES.

```sql
WITH sorted_seq AS (
    SELECT *
    FROM sequence s
    ORDER BY id
)
SELECT DISTINCT(s1.num)
FROM sorted_seq s1
    JOIN sorted_seq s2 ON s1.id + 1 = s2.id
    JOIN sorted_seq s3 ON s1.id + 2 = s3.id
WHERE
    s1.num = s2.num AND
    s2.num = s3.num
```