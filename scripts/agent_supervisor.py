import functools
import operator
import os
from typing import Annotated, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agent Supervisor"

tavily_tool = TavilySearchResults(max_results=5)
python_repl_tool = PythonREPLTool()

def agent_node(state, agent, name):
    """Process state through an agent and return updated state."""
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}

members = ["Researcher", "Coder"]
options = ["FINISH"] + members

class RouteResponse(BaseModel):
    """Response from supervisor agent."""
    next: Literal["FINISH", "Researcher", "Coder"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

class AgentState(TypedDict):
    """State for the multi-agent system."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

def supervisor_agent(state):
    """Supervisor agent that decides which worker should act next."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    supervisor_chain = prompt | llm.with_structured_output(RouteResponse)
    return supervisor_chain.invoke(state)

def create_supervisor_graph():
    """Create and configure the supervisor graph."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    research_agent = create_react_agent(llm, tools=[tavily_tool])
    research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")
    
    code_agent = create_react_agent(llm, tools=[python_repl_tool])
    code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

    workflow = StateGraph(AgentState)

    workflow.add_node("Researcher", research_node)
    workflow.add_node("Coder", code_node)
    workflow.add_node("supervisor", supervisor_agent)

    for member in members:
        workflow.add_edge(member, "supervisor")
    
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    workflow.add_edge(START, "supervisor")
    
    return workflow.compile()

def main():
    """Run the supervisor system with example queries."""
    graph = create_supervisor_graph()

    print("Example 1: Code Hello World")
    for s in graph.stream(
        {
            "messages": [
                HumanMessage(content="Code hello world and print it to the terminal")
            ]
        }
    ):
        if "__end__" not in s:
            print(s)
            print("----")

    print("\nExample 2: Research Report")
    for s in graph.stream(
        {"messages": [HumanMessage(content="Write a brief research report on pikas.")]},
        {"recursion_limit": 100},
    ):
        if "__end__" not in s:
            print(s)
            print("----")

if __name__ == "__main__":
    main() 