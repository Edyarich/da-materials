-- d7_buyer(id int, name text, last_action_date date);  
-- d7_seller(id int, name text);  
-- d7_manager(id int, name text);  
-- d7_user(id serial, role text, registration_date date);  
Найдите фамилию сотрудника с максимальной датой последнего действия

```
Ожидаемый формат ответа
+---------+
| surname |
+---------+
| Кузикин |
+---------+
```
**Ограничения**: запрещено использовать LIMIT, OVER.

```sql
select surname
from (
    select *
    from d7_buyer db 
    where last_action_date = (
        select max(last_action_date)
        from d7_buyer
    )
    union
    select *
    from d7_seller ds
    where last_action_date = (
        select max(last_action_date)
        from d7_seller
    )
    union
    select *
    from d7_manager dm
    where last_action_date = (
        select max(last_action_date)
        from d7_manager
    )
) as last_actions_info
where last_action_date = (
    select max(date)
    from (
        select max(last_action_date) as date
        from d7_buyer db
        union
        select max(last_action_date) as date
        from d7_seller ds
        union
        select max(last_action_date) as date
        from d7_manager dm
    ) as last_action_dates
)
```