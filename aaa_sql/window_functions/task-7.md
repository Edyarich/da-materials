-- create table d9_datamarts (dm varchar(64), calc_time int);  
-- create table d9_dag (src varchar(64), tgt varchar(4));  

Даны расчеты табличек в хранилище данных.

`d9_datamarts` - хранит название таблички и время в минутах которое необходимо для ее заполнения

`d9_dag` - хранит последовательность, в которой нужно заполнять таблички в виде пар.

Нужно вывести цепочку табличек которая заполняется дольше всего.

Цепочка состоит из названий таблиц, разделенных символами ' -> '

Например:
```
datamarts
dm	calc_time
A	9
B	6
C	1
D	8

dag
src	tgt
A	B
B	C
A	D
```
Самая долгая цепочка выглядит так: A -> D, она заполняется за 17 минут

Формат ответа:
```
calc_path	calc_time
A -> D	17
```

```sql
with recursive path_duration(start, next, path, p_time) as (
    select
        dm,
        dm,
        dm,
        calc_time
    from d9_datamarts
    union
    select
        pd.start,
        d.tgt,
        concat(pd.path, ' -> ', d.tgt),
        pd.p_time + dm.calc_time
    from path_duration pd
        join d9_dag d on d.src = pd.next
        join d9_datamarts dm on d.tgt = dm.dm
    
)
select
    path as calc_path,
    p_time as calc_time
from path_duration
order by p_time desc
limit 1
```