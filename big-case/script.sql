-- EDA
-- Начало эксперимента = 15.09.2021

-- В user_payments_info user_id -- это продавцы или покупатели?
-- Вывод запроса: 2.3 миллиона
-- Ответ: Продавцы
SELECT count(*)
FROM (
    SELECT user_id, item_id
    FROM user_payments_info
    INTERSECT
    SELECT user_id, item_id
    FROM user_item_info
) tbl


-- Вывод запроса: 7.7 миллиона
SELECT count(*)
FROM user_payments_info


-- Пересечение юзеров и айтемов между services_verification_discounts и user_payments_info
-- Вывод запроса: 20.3 k
SELECT count(*)
FROM (
    SELECT user_id, item_id
    FROM user_payments_info
    INTERSECT
    SELECT user_id, item_id
    FROM services_verification_discounts
) tbl


-- Вывод запроса: 30 k
SELECT count(*)
FROM services_verification_discounts


-- Проверка на `discount_day` > DATE('2021-09-15') в services_verification_discounts
-- Вывод запроса: 0
SELECT count(1)
FROM services_verification_discounts svd
WHERE discount_day = DATE('2021-09-15')


-- Последняя запись датируется вечером 01.12.2021
SELECT max(event_time)
FROM user_payments_info


-- Последняя запись датируется вечером 19.06.2022
SELECT max(item_creation_time)
FROM user_item_info



-- Итог: время завершения эксперимента = 02.12.2021
-- DATE('2021-12-02')


-- Сколько юзеров, участвующих в эксперименте, продали хоть 1 товар
-- Ответ: порядка 300к
SELECT count(1)
FROM (
    SELECT user_id
    FROM services_verification_experiment_segment
    GROUP BY user_id
    INTERSECT
    SELECT user_id
    FROM user_payments_info
    GROUP BY user_id
) tbl


-- Сравнение групп по региональному распределению айтемов
WITH active_users_and_discounts AS (
    SELECT
        sves.user_id,
        sves.experiment_group,
        sves."UserType",
        COALESCE(min(svd.discount_day), DATE('2021-09-15')) AS discount_day
    FROM services_verification_experiment_segment sves
        LEFT JOIN services_verification_discounts svd
            ON sves.user_id = svd.user_id
    GROUP BY
        sves.user_id,
        sves.experiment_group,
        sves."UserType"
)
SELECT
    item_region,
    count(item_region) AS items_count
FROM active_users_and_discounts aud
    JOIN user_item_info uii ON
        aud.user_id = uii.user_id AND
        uii.item_vertical = '1' AND
        uii.item_creation_time < aud.discount_day
WHERE aud.experiment_group = '60_discount'
GROUP BY item_region


-- Поскольку `active_users_and_discounts` будет часто фигурировать в запросах, создадим для нее VIEW 
CREATE VIEW active_users_and_discounts AS (
    SELECT
        sves.user_id,
        sves.experiment_group,
        sves."UserType",
        COALESCE(min(svd.discount_day), DATE('2021-09-15')) AS discount_day
    FROM services_verification_experiment_segment sves
        LEFT JOIN services_verification_discounts svd
            ON sves.user_id = svd.user_id
    GROUP BY
        sves.user_id,
        sves.experiment_group,
        sves."UserType"
)
    

-- Число скидочных объявлений по группам
SELECT
    experiment_group,
    count(1)
FROM active_users_and_discounts aud
WHERE discount_day > DATE('2021-09-15') 
GROUP BY experiment_group


-- Выручка по группам и чиcло проданных товаров до начала эксперимента (открыл и закрыл позицию ДО начала эксперимента)
SELECT
	uii.user_id,
    aud.experiment_group,
    aud."UserType",
    sum(upi.amount_net) AS user_revenue,
    count(uii.item_id) AS exposed_items
FROM active_users_and_discounts aud
    LEFT JOIN user_item_info uii ON
        aud.user_id = uii.user_id AND
        uii.item_creation_time < DATE('2021-09-15') AND
        uii.item_vertical = '1'
    LEFT JOIN user_payments_info upi ON
        aud.user_id = upi.user_id AND
        uii.item_id = upi.item_id AND
        upi.event_time < DATE('2021-09-15')
GROUP BY
	uii.user_id,
    aud.experiment_group,
    aud."UserType"
        

-- Выручка по группам во время эксперимента (открыл и закрыл позицию ПОСЛЕ начала эксперимента)
SELECT
	aud.user_id,
    aud.experiment_group,
    aud."UserType",
    CASE 
		WHEN min(aud.discount_day) = DATE('2021-09-15') THEN 0 ELSE 1
    END AS is_verified,
    sum(upi.amount_net) AS user_revenue,
    count(uii.item_id) AS exposed_items
FROM active_users_and_discounts aud
    JOIN user_item_info uii ON
        aud.user_id = uii.user_id AND
        uii.item_creation_time >= DATE('2021-09-15') AND
        uii.item_vertical = '1' AND
        uii.item_creation_time < DATE('2021-12-02')
    LEFT JOIN user_payments_info upi ON
        aud.user_id = upi.user_id AND
        uii.item_id = upi.item_id
GROUP BY
    aud.user_id,
    aud.experiment_group,
    aud."UserType"


-- Доли получивших значок по группам
SELECT
    experiment_group,
    "UserType",
    sum(has_sign) AS k_converted,
    count(has_sign) AS group_size,
    sum(has_sign) / count(has_sign) AS sign_conversion
FROM (
    SELECT
        *,
        CASE 
            WHEN discount_day = DATE('2021-09-15') THEN 0.0 ELSE 1.0
        END AS has_sign
    FROM active_users_and_discounts aud
) tbl
GROUP BY GROUPING SETS (
    (experiment_group),
    (experiment_group, "UserType")
)


-- DAU по группам во время эксперимента (активность = выложить товар)
SELECT
	uii.item_creation_time::date,
	aud.experiment_group,
	count(DISTINCT(aud.user_id)) AS k_active_users
FROM active_users_and_discounts aud
    JOIN user_item_info uii ON
        aud.user_id = uii.user_id AND
        uii.item_creation_time > DATE('2021-09-15') AND
        uii.item_vertical = '1' AND
        uii.item_creation_time < DATE('2021-12-02')
GROUP BY uii.item_creation_time::date, aud.experiment_group


-- Убытки от предоставленных скидок на продвижение
SELECT
    sves.experiment_group,
    sum(amount_net) AS received,
    sum(amount_net) * CASE sves.experiment_group
    	WHEN 'no_discount' THEN 0
    	WHEN '30_discount' THEN 1.0/0.7 - 1
    	WHEN '60_discount' THEN 1.0/0.4 - 1
    END AS costs
FROM services_verification_experiment_segment sves
    JOIN services_verification_discounts svd ON
        sves.user_id = svd.user_id
    JOIN user_item_info uii ON
        sves.user_id = uii.user_id AND
        svd.item_id = uii.item_id AND
        uii.item_creation_time > DATE('2021-09-15') AND
        uii.item_creation_time < DATE('2021-12-02') AND
        uii.item_vertical = '1'
    JOIN user_payments_info upi ON
    	sves.user_id = upi.user_id AND
    	svd.item_id = upi.item_id
GROUP BY
    sves.experiment_group

    
-- Проверка ожиданий
SELECT max(cnt)
FROM (
	SELECT
	    count(svd.item_id) AS cnt
	FROM services_verification_discounts svd
	GROUP BY
	    svd.user_id
) tbl

SELECT count(user_id)
FROM services_verification_experiment_segment
WHERE user_id % 2 = 0


