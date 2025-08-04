SELECT id, text 
FROM poi WHERE
SEMANTIC_FILTER_SINGLE_COL('Is {poi.text} a positive comment?', text) = 1;