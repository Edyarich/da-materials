**Тема**: агрегатные функции, группировки.

Дана таблица с описаниями объявлений:

```sql
create table item_descriptions (
    item_id int,       -- id объявления
    user_id int,       -- id пользователя, выложившего объявление
    descr varchar(127) -- описание объявления
);
```
**Задание**:

Посчитайте, сколько пользователей готовы торговаться (в описании присутствует фраза "торг уместен"), а сколько не готовы (фраза "без торга"). Не учитывайте объявления, из описания которых нельзя однозначно узнать о наличии торга.

Считаем, что:

- У пользователя может быть несколько объявлений;
- Если у одного пользователя есть объявления с торгом и без торга, то считаем, что он готов торговаться;
- Указанные фразы про торг могут присутствовать в описании в любом регистре;
- Указанные фразы про торг могут отсутствовать в описании объявления;
- Указанные фразы про торг не могут присутствовать одновременно в описании одного объявления.

**Пример**:

```
item_id|  user_id|  descr
-------|---------|----------------------------------
   1001|       10|  Диван отличный, без торга
   1002|       10|  Диван старый, торг уместен
   1003|       20|  Продам диван
   1004|       20|  Кресло с гарантией. Без торга!!
```

Ожидаемый формат ответа:
```
bidding     | user_cnt       
------------|----------
 С торгом   | 1
 Без торга  | 1
```
**Ограничения**: запрещено использовать связанные подзапросы (обращение в подзапросе к таблицам из внешнего запроса), LIMIT, UNION, UNION ALL, любые виды JOIN, VALUES, IN, OVER.