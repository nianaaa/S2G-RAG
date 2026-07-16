# bm25_search.py  —— 用 Pyserini 查询 BM25（完全替换本文件）

import pickle
from pyserini.search.lucene import LuceneSearcher as SimpleSearcher

# 载入之前保存的语料与检索设置
# with open('bm25_search/corpus/corpus.pkl', 'rb') as f:
#     corpus = pickle.load(f)

# with open('bm25_search/corpus/retriever_settings.pkl', 'rb') as f:
#     settings = pickle.load(f)
with open('bm25_search/corpus/2wiki_corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)

with open('bm25_search/corpus/2wiki_retriever_settings.pkl', 'rb') as f:
    settings = pickle.load(f)

def bm25_search(query_text, topk=10):
    index_dir = settings["index_dir"]
    k1 = settings.get("k1", 0.9)
    b = settings.get("b", 0.4)

    searcher = SimpleSearcher(index_dir)
    searcher.set_bm25(k1, b)

    hits = searcher.search(query_text, k=topk)

    print(f"Query : {query_text}")
    for rank, h in enumerate(hits, start=1):
        doc_id = h.docid
        title = corpus.get(doc_id, {}).get('title')
        text = corpus.get(doc_id, {}).get('text') or corpus.get(doc_id, {}).get('contents')
        print(f"Doc {rank}: {doc_id} [{title}] - {text}")

# 示例
if __name__ == "__main__":
    bm25_search("What is the capital of France?", topk=10)
