from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")


# JsonOutputParser → converts the LLM's JSON response
# into a Python dictionary
parser = JsonOutputParser()


template = PromptTemplate(
    template="""
    Give me the name, age and city of a famous Indian Cricketer.

    {format_instruction}
    """,

    # No normal input variables because the prompt
    # does not require user-provided values
    input_variables=[],

    # Injects instructions telling the LLM
    # what JSON format it should return
    partial_variables={
        'format_instruction': parser.get_format_instructions()
    }
)


# LCEL chain:
# Prompt → LLM → JSON Parser
chain = template | model | parser


# No input required because input_variables = []
response = chain.invoke({})

# response is now a Python dictionary
print(response)