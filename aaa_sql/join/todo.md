**Темы**: соединения таблиц, агрегатные функции.

Даны таблицы с информацией про заказы:
```sql
create table clients (
	order_id int,           -- id заказа
	client_id int,          -- id пользователя
);

create table goods (
	order_id int,            -- id заказа
	good_ids varchar(25)     -- id товаров
);

create table order_status (
	order_id int,             -- id заказа
        status_name varchar(25),  -- статус
	status_date date          -- дата
);
```
**Задание**:

Посчитайте долю доставленных заказов (status_name = 'delivered') от всех заказов у юзеров, у которых был созданный заказ (status_name = 'created') с товаром с good_id = 1. Ответ округлите до сотых.

**Пример**:
```
select * from clients;

order_id | client_id
---------|---------
 1       |  1
 3       |  1
 2       |  2
 4       |  1
select * from goods;

order_id | good_ids
---------|---------
 1       |  5,1,10
 3       |  6,1,8
 2       |  11
 4       |  11
select * from order_status;

order_id | status_name  | status_date
---------|--------------|------------
 1       | created      | 2022-01-01
 2       | created      | 2022-03-05
 2       | delivered    | 2022-04-01
 1       | declined     | 2022-01-01
 3       | created      | 2022-03-05
 4       | delivered    | 2022-01-05
```

Ожидаемый формат ответа:

```
share_orders
------------
0.33
```

**Ограничения**: запрещено использовать LIMIT, UNION, UNION ALL, VALUES, IN