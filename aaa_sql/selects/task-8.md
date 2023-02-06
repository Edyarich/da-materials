Дана таблица salary.
Найдите сотрудников с зарплатой, которая находится на третьем месте в топе зарплат по своему департаменту.

```
Ожидаемый формат ответа:
 department_id |  name   | salary 
---------------+---------+--------
             1 | Сотр_1  |  10000
             1 | Сотр_2  |  10000 
             2 | Сотр_3  |  50000
```

**Ограничения**: запрещено использовать HAVING, JOIN, LIMIT, UNION, UNION ALL, VALUES, оконные функции.

```
Query result:
+---------------+-----------+--------+
| department_id | name      | salary |
+---------------+-----------+--------+
| 2             | Дмитрий   | 55000  |
| 2             | Анастасия | 55000  |
| 1             | Илья      | 27500  |
| 3             | Григорий  | 55000  |
+---------------+-----------+--------+
Affected rows: 4
```

```sql
SELECT 
    department_id,
    name,
    salary
FROM salary
WHERE 
    (department_id, salary) IN (
        SELECT 
            department_id, 
            max(salary) AS third_max_dep_sal
        FROM salary
        WHERE (department_id, salary) NOT IN (
            SELECT 
                department_id, 
                max(salary) AS second_max_dep_sal
            FROM salary
            WHERE (department_id, salary) NOT IN (
                SELECT 
                    department_id, 
                    max(salary) AS max_dep_sal
                FROM salary
                GROUP BY department_id
            )
            GROUP BY department_id
        )
        AND (department_id, salary) NOT IN (
            SELECT 
                department_id, 
                max(salary) AS max_dep_sal
            FROM salary
            GROUP BY department_id
        )
        GROUP BY department_id
    )
```