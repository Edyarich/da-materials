-- d7_buyer(id int, name text, surname text, last_action_date date);  
-- d7_seller(id int, name text, surname text, last_action_date date);  
-- d7_manager(id int, name text, last_action_date date);  
-- d7_user(id serial, role text, registration_date date)  

Посчитайте количество пользователей в разрезе квартала регистрации и месяца.  
Подведите подытог по каждому разрезу  и общий подытог, отсортируйте по первой и второй колонке.

Формат ответа: 'q', 'm', 'cntd_users'

Синтаксис rollup в mysql:  GROUP BY <column>, ... WITH ROLLUP

Вычисление квартала и месяца: QUARTER(...), MONTH(...)  

```sql
select
    quarter(registration_date) as q,
    month(registration_date) as m,
    count(1) as cntd_users
from (
    select distinct id, registration_date
    from d7_user
) as unique_users
group by
    quarter(registration_date), 
    month(registration_date) with rollup
order by 1, 2
```