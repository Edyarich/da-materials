Даны таблицы клиентов, товаров и заказов. 

```sql
create table clients (
	client_id INTEGER PRIMARY KEY,
	name varchar(50)
);

create table goods (
	good_id INTEGER PRIMARY KEY,
	title varchar(50)
);

create table orders (
	order_id INTEGER PRIMARY KEY,
 	client_id INTEGER,
	good_id INTEGER
);
```
Нужно определить количество клиентов, которые не заказали ни одного товара, и количество товаров, которые никто не заказал.

```
Ожидаемый формат ответа:
 client_without_goods |  goods_without_client
----------------------+-----------------------
                 123  |                   234
```

**Ограничения**: запрещено использовать LIMIT, UNION, UNION ALL, VALUES, IN
+ В MySql нет full join, попробуйте решить без него

```sql
SELECT
    COUNT(DISTINCT c.client_id) - COUNT(DISTINCT o.client_id) AS client_without_goods,
    COUNT(DISTINCT g.good_id) - COUNT(DISTINCT o.good_id) AS goods_without_client
FROM orders o
    RIGHT JOIN clients c 
        ON 
            c.client_id = o.client_id
    RIGHT JOIN goods g 
        ON 
            g.good_id = o.good_id
            OR o.order_id IS NULL
```