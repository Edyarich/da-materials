Дана таблица Product:

```sql
CREATE TABLE product (
    model int primary key not null, 
    maker varchar(8),
    type varchar(16)
);
```

В таблице содержатся продукты нескольких типов: 'PC' - ПК, 'Laptop' - ПК-блокнот или 'Printer' - принтер. 

Найдите производителей, выпускающих ПК, но не ПК-блокноты.

```sql
SELECT DISTINCT maker
FROM product
WHERE 
    TYPE = 'PC'
    AND maker NOT IN (
        SELECT maker 
        FROM product
        WHERE 
            TYPE = 'Laptop'
    )
```