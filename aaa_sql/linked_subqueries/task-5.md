Даны таблички студентов, журнала с результатами решения задач и задачника.

```
registry_day4 (student_id int, subject varchar(16), task_no int, accepted boolean);
task_book_day4 (subject varchar(16), task_no int, difficulty int);
```

Вывести задачи по математике какой сложности были решены хотя бы одним студентом

(решить каким-то способом **in, any, exists**)

```
Формат ответа 
| difficulty |
| ---------- |
| 3          |
| 2          |
```

```sql
SELECT DISTINCT(tbd.difficulty)
FROM task_book_day4 tbd 
WHERE
    tbd.subject = 'math' AND
    EXISTS (
        SELECT rd.student_id
        FROM registry_day4 rd
        WHERE 
            rd.subject = tbd.subject AND
            rd.task_no = tbd.task_no AND
            rd.accepted = True
    )
```