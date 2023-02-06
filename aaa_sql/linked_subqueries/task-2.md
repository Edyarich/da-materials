Даны таблички студентов, журнала с результатами решения задач и задачника.

```
students_day4 (student_id int);
registry_day4 (student_id int, subject varchar(16), task_no int, accepted boolean);
task_book_day4 (subject varchar(16), task_no int, difficulty int);
```

Найти студентов, которые не решили правильно ни одной задачи по sql сложности 1
(решить каким-то способом **not exists, not in, left join**)

```
Формат ответа 
+------------+
| student_id |
+------------+
| 3          |
| 4          |
| 5          |
| 6          |
+------------+
```

```sql
SELECT sd.student_id
FROM students_day4 sd
    NATURAL LEFT JOIN registry_day4 rd
    LEFT JOIN task_book_day4 tbd ON
        rd.subject = tbd.subject AND
        rd.task_no = tbd.task_no AND
        tbd.difficulty = 1
WHERE COALESCE(rd.subject, 'sql') = 'sql'
GROUP BY sd.student_id
HAVING MAX(COALESCE(rd.accepted, 0) * COALESCE(tbd.difficulty, 0)) = 0
```