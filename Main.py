from dotenv import load_dotenv

from pydantic import BaseModel

from langchain_openai import ChatOpenAI

from langchain_anthropic import ChatAnthropic

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import PydanticOutputParser

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

from tools import search_tool, wiki_tool, save_tool



load_dotenv()
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        (
           "system",
            """
            You are the AI Research Agent – Agentic AI, a research assistant that helps generate research papers.

            Answer the user query and use neccessary tools. 

            Wrap the output in this format and provide no other text\n{format_instructions}

            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())
tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can i help you research? ")
raw_response = agent_executor.invoke({"query": query})
try:
    output = raw_response.get("output", "")
    if isinstance(output, list):
        output = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in output
        )
    structured_response = parser.parse(output)
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)
