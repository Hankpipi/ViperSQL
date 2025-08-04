SELECT id, text 
FROM poi WHERE
SEMANTIC_FILTER_SINGLE_COL('Is {poi.text} describing the service of a business?', text) = 1;