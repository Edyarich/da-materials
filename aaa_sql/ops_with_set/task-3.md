-- d7_buyer(id int, name text, surname text, last_action_date date);  
-- d7_seller(id int, name text, surname text, last_action_date date);  
-- d7_manager(id int, name text, last_action_date date);  
-- d7_user(id serial, role text, registration_date date);  
Для каждой роли подсчитайте максимальную дату последнего действия пользователей.

```
Ожидаемый формат ответа
+------+------------------+
| role | last_action_date |
+------+------------------+
| b    | 2021-05-18       |
| m    | 2021-05-30       |
| s    | 2021-04-12       |
+------+------------------+
```

```sql
select
    'b' as role,
    max(last_action_date) as last_action_date
from d7_buyer db 
union
select
    'm' as role,
    max(last_action_date) as last_action_date
from d7_manager dm
union
select
    's' as role,
    max(last_action_date) as last_action_date
from d7_seller ds
```