Отсортируйте таблицу cities (по столбцу city) следующим образом: в начале находится Санкт-Петербург, в конце находится Москва, а остальные города отсортированы в лексикографическом порядке по убыванию.

В select-листе нужно оставить только city.

```
Query result:
+-----------------+
| city            |
+-----------------+
| Санкт-Петербург |
| Нижневартовск   |
| Кострома        |
| Армавир         |
| Москва          |
+-----------------+
Affected rows: 5
```

```sql
SELECT city
FROM cities c
ORDER BY 
    city = 'Москва',
    city = 'Санкт-Петербург' DESC,
    city DESC;
```