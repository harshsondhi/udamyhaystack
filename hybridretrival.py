from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from datasets import load_dataset
from haystack import Document
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore

dataset = load_dataset("anakin87/medrag-pubmed-chunk", split="train")
docs = []
stop=0

for doc in dataset:
    if stop==1000:
        break
    stop = stop + 1
    docs.append(
        Document(content=doc["content"], meta={"title": doc["title"], "abstract": doc["content"],"pmid":doc["id"]})
    )
    
#print("Number of documents:", len(docs))    

document_store = InMemoryDocumentStore()
document_splitter = DocumentSplitter(split_length=512,split_by="word",split_overlap=32)
document_embedder = SentenceTransformersDocumentEmbedder()
document_embedder.warm_up()

document_writer=DocumentWriter(document_store=document_store)

index_pipeline = Pipeline()
index_pipeline.add_component("document_splitter", document_splitter)
index_pipeline.add_component("document_embedder", document_embedder)
index_pipeline.add_component("document_writer", document_writer)

index_pipeline.connect("document_splitter", "document_embedder")
index_pipeline.connect("document_embedder", "document_writer")
index_pipeline.run({"document_splitter": {"documents": docs}})

#step-3
text_embedder = SentenceTransformersTextEmbedder()
embedding_retriever = InMemoryEmbeddingRetriever(document_store=document_store)
mb25_retriever = InMemoryBM25Retriever(document_store=document_store)
document_joiner = DocumentJoiner()
ranker = TransformersSimilarityRanker()

hybrid_retriever = Pipeline()
hybrid_retriever.add_component("text_embedder", text_embedde)
hybrid_retriever.add_component("embedding_retriever", embedding_retriever)
hybrid_retriever.add_component("bm25_retriever", bm25_retriever)
hybrid_retriever.add_component("document_joiner", document_joiner)
hybrid_retriever.add_component("ranker", ranker)

hybrid_retriever.connect("text_embedder", "embedding_retriever")
hybrid_retriever.connect("bm25_retriever", "document_joiner")
hybrid_retriever.connect("embedding_retriever", "document_joiner")
hybrid_retriever.connect("document_joiner", "ranker")

hybrid_retriever.draw("hybrid_retriever.png")