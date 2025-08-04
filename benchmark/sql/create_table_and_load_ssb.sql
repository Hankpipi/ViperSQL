-- Enable LOCAL INFILE for the current session
SET GLOBAL local_infile = 1;
SET SESSION local_infile = 1;

DROP TABLE IF EXISTS customer;
DROP TABLE IF EXISTS ddate;
DROP TABLE IF EXISTS lineorder;
DROP TABLE IF EXISTS part;
DROP TABLE IF EXISTS supplier;

CREATE TABLE lineorder (
    lo_orderkey INTEGER NOT NULL,
    lo_linenumber INTEGER NOT NULL,
    lo_custkey INTEGER NOT NULL,
    lo_partkey INTEGER NOT NULL,
    lo_suppkey INTEGER NOT NULL,
    lo_orderdate INTEGER NOT NULL,
    lo_orderpriority TEXT NOT NULL,
    lo_shippriority TEXT NOT NULL,
    lo_quantity INTEGER NOT NULL,
    lo_extendedprice INTEGER NOT NULL,
    lo_ordtotalprice INTEGER NOT NULL,
    lo_discount INTEGER NOT NULL,
    lo_revenue INTEGER NOT NULL,
    lo_supplycost INTEGER NOT NULL,
    lo_tax INTEGER NOT NULL,
    lo_commitdate INTEGER NOT NULL,
    lo_shopmode TEXT NOT NULL,
    dummy DOUBLE,
    primary key (lo_orderkey,lo_linenumber)
);

CREATE TABLE part (
    p_partkey INTEGER NOT NULL,
    p_name TEXT NOT NULL,
    p_mfgr TEXT NOT NULL,
    p_category TEXT NOT NULL,
    p_brand1 TEXT NOT NULL,
    p_color TEXT NOT NULL,
    p_type TEXT NOT NULL,
    p_size INTEGER NOT NULL,
    p_container TEXT NOT NULL,
    dummy DOUBLE
);

CREATE TABLE supplier (
    s_suppkey INTEGER NOT NULL,
    s_name TEXT NOT NULL,
    s_address TEXT NOT NULL,
    s_city TEXT NOT NULL,
    s_nation TEXT NOT NULL,
    s_region TEXT NOT NULL,
    s_phone TEXT NOT NULL,
    dummy DOUBLE
);

CREATE TABLE customer (
    c_custkey INTEGER NOT NULL,
    c_name TEXT NOT NULL,
    c_address TEXT NOT NULL,
    c_city TEXT NOT NULL,
    c_nation TEXT NOT NULL,
    c_region TEXT NOT NULL,
    c_phone TEXT NOT NULL,
    c_mktsegment TEXT NOT NULL,
    dummy DOUBLE
);

CREATE TABLE ddate (
    d_datekey INTEGER NOT NULL,
    d_date TEXT NOT NULL,
    d_dayofweek TEXT NOT NULL,
    d_month TEXT NOT NULL,
    d_year INTEGER NOT NULL,
    d_yearmonthnum INTEGER NOT NULL,
    d_yearmonth TEXT NOT NULL,
    d_daynuminweek INTEGER NOT NULL,
    d_daynuminmonth INTEGER NOT NULL,
    d_daynuminyear INTEGER NOT NULL,
    d_monthnuminyear INTEGER NOT NULL,
    d_weeknuminyear INTEGER NOT NULL,
    d_sellingseasin TEXT NOT NULL,
    d_lastdayinweekfl INTEGER NOT NULL,
    d_lastdayinmonthfl INTEGER NOT NULL,
    d_holidayfl INTEGER NOT NULL,
    d_weekdayfl INTEGER NOT NULL,
    dummy DOUBLE,
    primary key (d_datekey)
);
 
-- Data loading using LOCAL
LOAD DATA LOCAL INFILE '/home/zihao/ViperSQL/benchmark/data/storage/sf16_lineorder.csv'
INTO TABLE lineorder
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES;

LOAD DATA LOCAL INFILE '/home/zihao/ViperSQL/benchmark/data/storage/sf16_part.csv'
INTO TABLE part
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES;

LOAD DATA LOCAL INFILE '/home/zihao/ViperSQL/benchmark/data/storage/sf16_supplier.csv'
INTO TABLE supplier
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES;

LOAD DATA LOCAL INFILE '/home/zihao/ViperSQL/benchmark/data/storage/sf16_customer.csv'
INTO TABLE customer
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES;

LOAD DATA LOCAL INFILE '/home/zihao/ViperSQL/benchmark/data/storage/sf16_date.csv'
INTO TABLE ddate
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES;
