Дана табличка клиентов.

```sql
CREATE TABLE customer (
	id int4 NOT NULL,
	first_name varchar(50) NULL,
	last_name varchar(50) NULL,
	city varchar(50) NULL,
);
```

Нужно вывести пары клиентов, которые проживают в одном городе. Выведите каждую пару только 1 раз.

```
Формат ответа 
| id1     | id2    |
| ------- | ------ |
| 1       | 10     |
| 2       | 18     |
```

```sql
SELECT 
    left_c.id AS id1, 
    right_c.id AS id2
FROM customer left_c
    JOIN customer right_c
    ON 
        left_c.id < right_c.id
        AND left_c.city = right_c.city
```