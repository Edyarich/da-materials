Объясните, что не так с запросом, что может ввести в заблуждение?

```
select table1.foo, table2.foo, table2.bar
from table1
left join table2
on table1.foo = table2.foo
where table2.bar = 5
```

Если все в порядке, то опишите, что по-вашему он делает.

```
Смущает тот факт, что обычно указывают: **from table_name t1 left join oth_table_name t2 on ...** 

Плюс ко всему, применяется условие сравнения к данным из второго столбца, которые могут быть равны NULL ввиду left join-а

Также непонятен смысл применения join-а, поскольку в условном выражении участвует только столбец из второй таблицы. То есть, можно было просто написать запрос с условием ко второй таблице. 

Запрос выведет данные формата: **(id, id, 5)**
```