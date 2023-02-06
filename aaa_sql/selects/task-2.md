Дана таблица с описаниями объявлений:

```sql
CREATE TABLE item_descriptions (
    item_id INTEGER PRIMARY KEY,
    user_id INTEGER, 
    descr VARCHAR(127)
);
```

Выведете все id объявлений (item_id), у которых в поле с описанием есть 2 подряд идущих символа '#' или 2 подряд идущих символа '/'.

```sql
SELECT item_id
FROM item_descriptions id 
WHERE
    descr LIKE '%##%'
    OR descr LIKE '%//%';
```