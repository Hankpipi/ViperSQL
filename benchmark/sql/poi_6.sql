SELECT content, topic
FROM poi, topic
WHERE SEM_JOIN( 'Is {poi.content} relevant to {topic.topic}?', poi.content, topic.topic);