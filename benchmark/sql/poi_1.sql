SELECT id, text 
FROM poi WHERE
SEMANTIC_FILTER_SINGLE_COL('Is {poi.text} describing a restaurant?', text) = 1;