from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent,AgentExecutor
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
agent=create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=[]
)

agent_executor=AgentExecutor(agent=agent,tools=[],verbose=True)


raw_response=agent_executor.invoke({"query":"What is the capital of Nigeria?"})
print(raw_response)

structured_response=parser.parse(raw_response.get("output")[0]['text']) 
print(structured_response)
