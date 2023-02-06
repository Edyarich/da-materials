Даны две таблицы:

- таблица с заказами - orders 
- таблица со статусами заказов - order_status
- 
Выведите все заказы за 1 марта 2020, у которых статус Open или же их нет в таблице статусов.

```
Ожидаемый формат ответа:
order_id
---------
 C0003
 C0004
```

**Ограничения**: запрещено использовать связанные подзапросы (обращение в подзапросе к таблицам из внешнего запроса), LIMIT, UNION, UNION ALL, VALUES

```sql
SELECT o.id as order_id
FROM orders o 
    LEFT JOIN order_status os ON o.id = os.id
WHERE 
    o.order_date = '2020-03-01'
    AND (os.status = 'Open'
    OR os.status IS NULL)
```