from typing import List
from haystack import component, Document, Pipeline


@component
class WelcomeTextGenerator:
    """ 
    This component is used to generate a welcome text for the user and make it upper case.
    """
    @component.output_types(welcome_text=str, note=str)
    def run(self, user_name: str) :
        return {
            "welcome_text": ("Welcome {user_name} to Haystack!".format(user_name=user_name)).upper(),
            "note": "Welcome message is ready",
        }
        
        
@component        
class WhitespaceSplitter:
    """ 
    This component is used to split the text into words based on whitespace.
    """
    @component.output_types(splited_text=List[str])
    def run(self, text: str):
        return {"splited_text":text.split() }    
    
    
    
text_pipeline = Pipeline()
text_pipeline.add_component('welcome_text_generator', WelcomeTextGenerator())
text_pipeline.add_component('whitespace_splitter', WhitespaceSplitter())

text_pipeline.connect("welcome_text_generator.welcome_text", "whitespace_splitter.text") 

result = text_pipeline.run({"welcome_text_generator": {"user_name": "John"}})

print(result['whitespace_splitter']['splited_text'])
       