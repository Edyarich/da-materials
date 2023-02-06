-- d7_buyer(id int, name text, surname text, last_action_date date);  
-- d7_seller(id int, name text, surname text, last_action_date date);  
-- d7_manager(id int, name text, surname text, last_action_date date);  
-- d7_user(id serial, role text, registration_date date);  

Найдите месяцы в которых было более трех последний действий покупателей и более двух последних действий продавцов

```
Ожидаемый формат ответа
+------------+
| month      |
+------------+
| 2020-11-01 |
+------------+
```

К началу месяца дату можно привести, например, так 

DATE_FORMAT(last_action_date, '%Y-%m-01')

(Оператора intersect в mysql нет)

```sql
select buyer_dates.month
from (
    select date_format(last_action_date, '%Y-%m-01') as month
    from d7_buyer db
    group by date_format(last_action_date, '%Y-%m-01')
    having count(1) > 3
) as buyer_dates
join (
    select date_format(last_action_date, '%Y-%m-01') as month
    from d7_seller ds
    group by date_format(last_action_date, '%Y-%m-01')
    having count(1) > 2
) as seller_dates
    on buyer_dates.month = seller_dates.month
```