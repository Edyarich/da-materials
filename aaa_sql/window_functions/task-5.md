-- create table d9_employee (id int, name text, manager_id int, salary int)

Для каждого менеджера первого уровня найдите долю, которую составляет его зарплата и зарплата всех его подчиненных от общих трат на зарплату

Ответ округлите до целых с помощью функции ROUND

```
Формат ответа
+----------------+-----------------------+
| top_manager_id | department_salary_pct |
+----------------+-----------------------+
| 1              | 56                    |
| 2              | 22                    |
| 3              | 22                    |
+----------------+-----------------------+
```

```sql
with recursive dep_total(id, main_parent_id, parent_id, dep_salary) as (
    select
        id,
        id,
        manager_id,
        salary
    from d9_employee
    where manager_id is null
    union
    select
        de.id,
        dt.main_parent_id,
        de.manager_id,
        de.salary
    from dep_total as dt
        join d9_employee de on dt.id = de.manager_id
)
select distinct
    main_parent_id as top_manager_id,
    round(
        100 * sum(dep_salary) over (partition by main_parent_id) / sum(dep_salary) over ()
    ) as department_salary_pct
from dep_total
```