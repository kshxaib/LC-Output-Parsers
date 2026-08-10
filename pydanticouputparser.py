from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()


model = ChatOpenAI(model="gpt-3.5-turbo")

class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")
    occupation: str = Field(description="The occupation of the person")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = "Generate the name, age and occupation of a famous person from {country}. \n{format_instructions}",
    input_variables = ["country"],
    partial_variables = {"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser

response = chain.invoke({"country": "Pakistan"})

print(response)