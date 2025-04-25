import functools
import operator
import os
from typing import Annotated, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Multi-agent Collaboration"

def create_agent(llm, agent_tools, system_message: str):
    """Create an agent with specific tools and system message."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([agent_tool.name for agent_tool in agent_tools]))
    return prompt | llm.bind_tools(agent_tools)

tavily_tool = TavilySearchResults(max_results=5)
repl = PythonREPL()

@tool
def python_repl(code: str):
    """Execute python code to generate charts or perform calculations."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."

class AgentState(TypedDict):
    """State for the multi-agent system."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

def agent_node(state, agent, name):
    """Process state through an agent and return updated state."""
    result = agent.invoke(state)
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.model_dump(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        "sender": name,
    }

def router(state) -> Literal["call_tool", "__end__", "continue"]:
    """Route messages between agents and tools."""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        return "__end__"
    return "continue"

def create_multi_agent_graph():
    """Create and configure the multi-agent graph."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    research_agent = create_agent(
        llm,
        [tavily_tool],
        system_message="You should provide accurate data for the chart_generator to use.",
    )
    research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")
    
    chart_agent = create_agent(
        llm,
        [python_repl],
        system_message="Any charts you display will be visible by the user.",
    )
    chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_generator")

    tools = [tavily_tool, python_repl]
    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentState)

    workflow.add_node("Researcher", research_node)
    workflow.add_node("chart_generator", chart_node)
    workflow.add_node("call_tool", tool_node)

    workflow.add_conditional_edges(
        "Researcher",
        router,
        {"continue": "chart_generator", "call_tool": "call_tool", "__end__": END},
    )
    workflow.add_conditional_edges(
        "chart_generator",
        router,
        {"continue": "Researcher", "call_tool": "call_tool", "__end__": END},
    )
    workflow.add_conditional_edges(
        "call_tool",
        lambda x: x["sender"],
        {
            "Researcher": "Researcher",
            "chart_generator": "chart_generator",
        },
    )
    workflow.add_edge(START, "Researcher")
    
    return workflow.compile()

def main():
    """Run the multi-agent system with an example query."""
    graph = create_multi_agent_graph()

    print("Example: UK GDP Analysis")
    events = graph.stream(
        {
            "messages": [
                HumanMessage(
                    content="Fetch the UK's GDP over the past 5 years,"
                    " then draw a line graph of it."
                    " Once you code it up, finish."
                )
            ],
        },
        {"recursion_limit": 150},
    )
    
    for s in events:
        print(s)
        print("----")

if __name__ == "__main__":
    main() 