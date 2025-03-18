from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent,AgentExecutor
from tools import search_tool,wiki_tool,save_tool
import streamlit as st
import os

load_dotenv()
os.environ['ANTHROPIC_API_KEY']=os.getenv("ANTHROPIC_API_KEY")

class ResearchResponse(BaseModel):
    topic:str
    summary:str
    sources:list[str]
    tools_used:list[str]





llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# response=llm.invoke("explain llm")
# print(response)
parser=PydanticOutputParser(pydantic_object=ResearchResponse)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# create agent
tools=[search_tool,wiki_tool,save_tool]
agent=create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

st.title("Research Assistant")
query=st.text_input("Search the topic you want to research")


# raw_response=agent_executor.invoke({"query":"What is the capital of Nigeria?"})
# # print(raw_response)
if query:
    raw_response=agent_executor.invoke({"query":query})

    try:

        structured_response=parser.parse(raw_response.get("output")[0]['text'])
        st.write(structured_response)

    except Exception as e:
        st.write(f"Error parsing response: {e} Raw response: {raw_response}")