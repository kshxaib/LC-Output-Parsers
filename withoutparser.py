from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)


template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables=['text']
)


prompt1 = template1.invoke({'topic': "AI"})
response1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text': response1.content})
response2 = model.invoke(prompt2)

print(response1.content)
print("-------------------------------------------------------")
print(response2.content)