from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")


# Define the structure we want from the LLM
response_schema = [
    ResponseSchema(
        name="fact_1",
        description="Fact 1 about the topic"
    ),
    ResponseSchema(
        name="fact_2",
        description="Fact 2 about the topic"
    ),
    ResponseSchema(
        name="fact_3",
        description="Fact 3 about the topic"
    )
]


# Create StructuredOutputParser from the schema
parser = StructuredOutputParser.from_response_schemas(response_schema)


# PromptTemplate
template = PromptTemplate(
    template="""
    Give 3 facts about the {topic}.

    {format_instructions}
    """,

    input_variables=["topic"],

    # Automatically generate formatting instructions
    # and insert them into the prompt
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)


# LCEL chain:
# Prompt → LLM → StructuredOutputParser
chain = template | model | parser


# Run the chain
result = chain.invoke({
    "topic": "AI"
})

print(result)