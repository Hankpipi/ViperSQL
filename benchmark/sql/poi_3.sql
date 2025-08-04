SELECT id, text 
FROM poi WHERE
SEMANTIC_FILTER_SINGLE_COL('Is {poi.text} describing Japenese food?', text) = 1;