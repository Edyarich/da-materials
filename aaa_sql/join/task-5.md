Даны таблички

```sql
CREATE TABLE h_item (item_id int);
CREATE TABLE s_item_price (
	item_id int,
	price int,
	actual_date timestamp
);
CREATE TABLE s_item_title (
	item_id int,
	title varchar(64),
	actual_date timestamp
);
```
В h_item есть все id объявлений.  
При изменении цены или названия объявления - в соответствующую табличку добавляется новая строка с текущей датой.  
Нужно вывести актуальные цену и название по каждому item_id. Если цены нет - выдайте 0, если нет названия - прочерк "-".

```
Формат ответа
+---------+-------+-----------+
| item_id | price | title     |
+---------+-------+-----------+
| 1       | 25000 | iphone X  |
| 2       | 0     | new chair |
| 3       | 1000  | metal box |
| 4       | 100   | -         |
+---------+-------+-----------+
```
Если вы решили задачу правильно - подумайте, как бы вы решали эту задачу, если бы у вас не было таблички h_item.
Попробуйте написать запрос в нашей учебной базе.

**Ограничения**: запрещено использовать связанные подзапросы (обращение в подзапросе к таблицам из внешнего запроса), LIMIT, UNION, UNION ALL, VALUES, IN.

```sql
SELECT 
    hi.item_id,
    COALESCE(price, 0) AS price,
    COALESCE(title, '-') AS title
FROM h_item hi
    NATURAL LEFT JOIN (
        SELECT 
            item_id, 
            MAX(actual_date) AS price_date
        FROM s_item_price sip
        GROUP BY item_id
    ) prices_dates
    NATURAL LEFT JOIN (
        SELECT 
            item_id, 
            MAX(actual_date) AS title_date
        FROM s_item_title sit
        GROUP BY item_id
    ) titles_dates
    LEFT JOIN s_item_price sip
        ON
            hi.item_id = sip.item_id
            AND prices_dates.price_date = sip.actual_date 
    LEFT JOIN s_item_title sit
        ON
            hi.item_id = sit.item_id
            AND titles_dates.title_date = sit.actual_date
```
