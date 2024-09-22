from haystack.components.embedders import SentenceTransformersTextEmbedder

text_to_embed = "I love pizza"
text_embedder = SentenceTransformersTextEmbedder()
text_embedder.warm_up()
print(text_embedder.run(text_to_embed))