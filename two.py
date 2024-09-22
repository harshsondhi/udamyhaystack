
from haystack import Document
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

documents = [Document(content="My name is Wolfgang and I live in Berlin"),
             Document(content="I saw a black horse running"),
             Document(content="Germany has many big cities")]

document_embedder = SentenceTransformersDocumentEmbedder()
document_embedder.warm_up()
document_with_embeddings = document_embedder.run(documents)['documents']


document_store = InMemoryDocumentStore()
document_store.write_documents(document_with_embeddings )

query_pipeline = Pipeline()
query_pipeline.add_component('text_embedder', SentenceTransformersTextEmbedder())
query_pipeline.add_component('retriever', InMemoryEmbeddingRetriever(document_store=document_store))

query_pipeline.connect("text_embedder.embedding","retriever.query_embedding")

query = "Who lives in Derlin"

result = query_pipeline.run({"text_embedder":{"text":query}})
print(result['retriever']['documents'][0])