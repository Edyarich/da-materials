Дана таблица с данными студентов:

```sql
CREATE TABLE students (
  student_id int4 NULL,
  name varchar(256) NULL,
  group_code varchar(256) NULL,
  birthday date NULL,
  studentship numeric(15, 2) NULL
);
```

Выведете имена и год рождения тех студентов, которые подходят под описание:

- имя начинается на букву А и год рождения строго меньше 1999
- имя начинается на букву П и год рождения строго больше 1999

Можно считать, что формат заполнения поля `name` - "Фамилия Имя[ Отчество]",
т.е.  поле name обязательно содержит фамилию и имя, а отчество может отсутствовать.

Ответ отсортируйте по student_id.
```
Query result:
+----------------------------+------+
| name                       | year |
+----------------------------+------+
| Иванов Александр           | 1990 |
| Иванова Анастасия          | 1998 |
| Иванов Алекс Николаевич    | 1990 |
| Иванова Аня Николаевна     | 1998 |
| Сидоров Петр               | 2000 |
| Сидорова Полина            | 2000 |
| Николаев Петр Николаевич   | 2000 |
| Петров Антон Александрович | 1998 |
| Петров Петр Петрович       | 2005 |
+----------------------------+------+
```

```sql
SELECT 
    name, 
    EXTRACT(YEAR FROM birthday) AS year
FROM students s 
WHERE 
    (name LIKE '% А%'
    AND (name NOT LIKE '% % А%'
    OR name LIKE '% А% А%')
    AND EXTRACT(YEAR FROM birthday) < 1999)
    OR (name LIKE '% П%'
    AND (name NOT LIKE '% % П%'
    OR name LIKE '% П% П%')
    AND EXTRACT(YEAR FROM birthday) > 1999)
ORDER BY student_id;
```