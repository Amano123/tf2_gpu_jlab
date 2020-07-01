from elasticsearch import Elasticsearch

def setting():
    settings ={
        "settings": {
            "analysis": {
                "tokenizer": {
                    "sudachi_tokenizer": {
                        "type": "sudachi_tokenizer",
                        "discard_punctuation": True,
                        "resources_path": "/usr/share/elasticsearch/config/sudachi",
                        "settings_path": "/usr/share/elasticsearch/config/sudachi/sudachi_fulldict.json"
                    }
                },
                "analyzer": {
                    "sudachi_analyzer": {
                        "filter": [
                        ],
                        "tokenizer": "sudachi_tokenizer",
                        "type": "custom"
                    }
                }
            }
        }
    }
    return settings

jp_index = "test"
es = Elasticsearch("elasticsearch-sudachi", use_ssl=False, verify_certs = False)
if es.indices.exists(index = jp_index):
    print(f"{jp_index}を更新します。")
    es.indices.delete(index = jp_index)

es.indices.create(index="test", body=setting())
