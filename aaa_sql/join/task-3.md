**credit** (log_id, acc_id, date, amnt) транзакции списания со счетов  
**debit** (log_id, acc_id, date, amnt) транзакции пополнения счетов

Нужно подготовить запрос, который выводит счет, дату, и как изменилась сумма на счете за эту дату, если изменения были (за 1 день можно быть несколько списаний и пополнений)

Пример:

```
credit
1, 1000197, 2021-01-02, 50
2, 1000197, 2021-01-03, 150
3, 1000197, 2021-01-05, 500

debit
1, 1000197, 2021-01-01, 100
2, 1000197, 2021-01-02, 200
3, 1000197, 2021-01-02, 300

Ответ
acc_id, date, diff
1000197, 2021-01-01, 100
1000197, 2021-01-02, 450
1000197, 2021-01-03, -150
1000197, 2021-01-05, -500
```

```sql
SELECT 
    COALESCE(debit_per_day.acc_id, credit_per_day.acc_id) AS acc_id,
    COALESCE(debit_per_day.date, credit_per_day.date) AS date,
    COALESCE(debit, 0) - COALESCE(credit, 0) AS diff
FROM (
    SELECT 
        acc_id, 
        date, 
        sum(amnt) AS debit
    FROM debit d 
    GROUP BY 
        acc_id, 
        date 
) debit_per_day
    FULL JOIN (
        SELECT 
            acc_id, 
            date, 
            sum(amnt) AS credit
        FROM credit c 
        GROUP BY 
            acc_id, 
            date 
    ) credit_per_day 
    ON 
        debit_per_day.acc_id = credit_per_day.acc_id 
        AND debit_per_day.date = credit_per_day.date
ORDER BY 
    acc_id,
    date
```