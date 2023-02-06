Дана таблица PC:

```sql
CREATE TABLE pc (
    code varchar(8) PRIMARY KEY,
    model int, 
    speed int,
    cd varchar(8),
    hd int,
    price int
);
```

Найдите номер модели, скорость и размер жесткого диска ПК, имеющих 12x или 24x CD и цену менее 600 дол.

```sql
SELECT 
    model, 
    speed, 
    hd
FROM pc p 
WHERE TRUE 
    AND cd IN ('12x', '24x')
    AND price < 600;
```