import os
from haystack import Pipeline,Document

from haystack.utils import Secret
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder 



from dotenv import load_dotenv
_ = load_dotenv() 

documents = [Document(content="My name is Harsh and I live in Torrance, California USA"),Document(content="i am a software engineer")]

prompt_template = """ 
    Given these documents, answer the following questions, \nDocuments:
    {% for document in documents %}
        {{document.content}}
    {% endfor %}
    
    \nQuestion: {{question}}
    \nAnswer: 

"""

p = Pipeline()
p.add_component(instance=PromptBuilder(template=prompt_template),name="prompt_builder")
p.add_component(instance=OpenAIGenerator(),name="generator")
p.connect("prompt_builder","generator")

question = "Where do I live?"

result = p.run({"prompt_builder":{"documents":documents,"question":question}})
print(result.get("generator").get("replies"))

