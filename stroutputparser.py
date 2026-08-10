from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")


# Prompt 1: Generate a detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)


# Prompt 2: Summarize the generated report
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. \n {text}",
    input_variables=['text']
)


# Converts the LLM's response into a plain Python string
parser = StrOutputParser()


# LCEL (LangChain Expression Language) chain:
# 1. template1 → creates report prompt
# 2. model → generates detailed report
# 3. parser → converts AIMessage into string
# 4. template2 → uses that string as {text}
# 5. model → generates summary
# 6. parser → converts final response into string
chain = template1 | model | parser | template2 | model | parser


# Input goes to {topic} in template1
response = chain.invoke({'topic': 'AI'})

print(response)