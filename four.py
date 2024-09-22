from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy



documets = [
    Document(id=1,content="This is document 1"),
    Document(id=2,content="This is document 2"),
]

document_store = InMemoryDocumentStore()

embedder = SentenceTransformersDocumentEmbedder()
document_writer = DocumentWriter(document_store=document_store,policy=DuplicatePolicy.NONE)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("embedder",embedder)
indexing_pipeline.add_component("document_writer",document_writer)
indexing_pipeline.connect("embedder","document_writer")

results = indexing_pipeline.run({"embedder":{"documents":documets}})

print(results)