Дата таблички с сотрудниками и зарплатами. Найти сотрудников, которые находятся на 3-ем месте по размеру зарплаты. Поле зарплаты может быть незаполнено.

```sql
CREATE TABLE employee (
	id int,
	name varchar(16),
	salary int4
);
```

Вывести имя и запрату.

```
Формат ответа
| name | salary |
| ---- | ------ |
| Alex | 30000  |
| Olga | 30000  |
```

**Ограничения**: запрещено использовать UNION, UNION ALL, VALUES, JOIN, CONSTANTS, MAX, MIN

```sql
SELECT
    name,
    salary
FROM employee e1
WHERE salary = (
    SELECT DISTINCT(salary) AS uniq_salary
    FROM employee e2
    ORDER BY uniq_salary DESC
    LIMIT 1
    OFFSET 2
)
```