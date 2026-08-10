from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age and city of a famous Indian Cricketer. /n {farmat_instruction}",
    input_variables=[],
    partial_variables = {'farmat_instruction': parser.get_format_instructions()}
)

# prompt = template.format()
# response = model.invoke(prompt)
# output = parser.parse(response.content)

chain = template | model | parser
response = chain.invoke({})

print(response)