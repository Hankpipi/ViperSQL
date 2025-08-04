SET GLOBAL local_infile = 1;
SET SESSION rocksdb_bulk_load = ON;

CREATE TABLE `poi` (
  `coordinate` point NOT NULL SRID 4326,
  `id` int NOT NULL,
  `review_id` varchar(30) NOT NULL,
  `text` text NOT NULL,
  `business_id` varchar(30) NOT NULL,
  `stars` float NOT NULL,
  `date` timestamp NOT NULL,
  `user_id` varchar(30) NOT NULL,
  `city_id` int NOT NULL,
  `text_embedding` json NOT NULL FB_VECTOR_DIMENSION 128
) ENGINE=ROCKSDB;


LOAD DATA LOCAL INFILE '/usr/local/share/data/poi.csv'
INTO TABLE poi
FIELDS
  TERMINATED BY ','
  OPTIONALLY ENCLOSED BY '"'
  ESCAPED BY '\\'
LINES
  TERMINATED BY '\n'
IGNORE 1 ROWS
(@id,
 @review_id,
 @txt,
 @business_id,
 @stars,
 @dt,
 @user_id,
 @coord,
 @city_id,
 @embedding)
SET
  id             = @id,
  review_id      = @review_id,
  `text`         = @txt,
  business_id    = @business_id,
  stars          = @stars,
  `date`         = STR_TO_DATE(@dt, '%Y-%m-%d %H:%i:%s'),
  user_id        = @user_id,
  coordinate     = ST_GeomFromText(@coord, 4326),
  city_id        = @city_id,
  text_embedding = CAST(@embedding AS JSON);
