import functools
import operator
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Hierarchical Agent Teams"
os.environ["USER_AGENT"] = "HierarchicalAgentTeams/1.0"

_TEMP_DIRECTORY = TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)

tavily_tool = TavilySearchResults(max_results=5)

@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )

@tool
def create_outline(points: List[str], file_name: str) -> str:
    """Create and save an outline."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"Outline saved to {file_name}"

@tool
def read_document(file_name: str, start: Optional[int] = None, end: Optional[int] = None) -> str:
    """Read the specified document."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    if start is not None:
        start = 0
    return "\n".join(lines[start:end])

@tool
def write_document(content: str, file_name: str) -> str:
    """Create and save a text document."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.write(content)
    return f"Document saved to {file_name}"

@tool
def edit_document(file_name: str, inserts: Dict[int, str]) -> str:
    """Edit a document by inserting text at specific line numbers."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()

    sorted_inserts = sorted(inserts.items())
    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."

    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)
    return f"Document edited and saved to {file_name}"

@tool
def python_repl(code: str):
    """Execute python code to generate charts or perform calculations."""
    try:
        result = PythonREPL().run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"

def agent_node(state, agent, name):
    """Process state through an agent and return updated state."""
    try:
        result = agent.invoke(state)
        if not isinstance(result, dict) or "messages" not in result:
            raise ValueError(f"Agent {name} returned invalid result format: {result}")
        return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}
    except Exception as e:
        print(f"Error in agent {name}: {e}")
        return {
            "messages": [
                HumanMessage(
                    content=f"Error occurred in {name}: {str(e)}",
                    name=name
                )
            ]
        }

def create_team_supervisor(llm: ChatOpenAI, system_prompt: str, members: List[str]):
    """Create a supervisor agent for a team."""
    options = ["FINISH"] + members
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}"
                "\nRespond with ONLY the name of the next role or FINISH.",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    
    def parse_output(message) -> dict:
        """Parse the output to get the next role."""
        if hasattr(message, 'content'):
            output = message.content.strip()
        else:
            output = str(message).strip()
            
        if output not in options:
            print(f"Warning: Invalid output '{output}', defaulting to FINISH")
            return {"next": "FINISH"}
        return {"next": output}
    
    chain = prompt | llm | parse_output
    return chain

def create_research_team():
    """Create the research team graph."""
    class ResearchTeamState(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
        team_members: List[str]
        next: str

    llm = ChatOpenAI(model="gpt-4", temperature=0)

    search_agent = create_react_agent(llm, tools=[tavily_tool])
    search_node = functools.partial(agent_node, agent=search_agent, name="Search")
    
    research_agent = create_react_agent(llm, tools=[scrape_webpages])
    research_node = functools.partial(agent_node, agent=research_agent, name="WebScraper")
    
    supervisor_agent = create_team_supervisor(
        llm,
        "You are a supervisor tasked with managing a conversation between the"
        " following workers:  Search, WebScraper. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH.",
        ["Search", "WebScraper"],
    )

    research_graph = StateGraph(ResearchTeamState)
    research_graph.add_node("Search", search_node)
    research_graph.add_node("WebScraper", research_node)
    research_graph.add_node("supervisor", supervisor_agent)

    research_graph.add_edge("Search", "supervisor")
    research_graph.add_edge("WebScraper", "supervisor")
    research_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {"Search": "Search", "WebScraper": "WebScraper", "FINISH": END},
    )
    research_graph.add_edge(START, "supervisor")
    
    return research_graph.compile()

def create_doc_writing_team():
    """Create the document writing team graph."""
    class DocWritingState(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
        team_members: str
        next: str
        current_files: str

    def prelude(state):
        written_files = []
        if not WORKING_DIRECTORY.exists():
            WORKING_DIRECTORY.mkdir()
        try:
            written_files = [
                f.relative_to(WORKING_DIRECTORY) for f in WORKING_DIRECTORY.rglob("*")
            ]
        except (FileNotFoundError, PermissionError, OSError) as e:
            print(f"Warning: Could not list files in working directory: {e}")
            pass
        if not written_files:
            return {**state, "current_files": "No files written."}
        return {
            **state,
            "current_files": "\nBelow are files your team has written to the directory:\n"
            + "\n".join([f" - {f}" for f in written_files]),
        }

    llm = ChatOpenAI(model="gpt-4", temperature=0)

    doc_writer_agent = create_react_agent(llm, tools=[write_document, edit_document, read_document])
    context_aware_doc_writer_agent = prelude | doc_writer_agent
    doc_writing_node = functools.partial(agent_node, agent=context_aware_doc_writer_agent, name="DocWriter")
    
    note_taking_agent = create_react_agent(llm, tools=[create_outline, read_document])
    context_aware_note_taking_agent = prelude | note_taking_agent
    note_taking_node = functools.partial(agent_node, agent=context_aware_note_taking_agent, name="NoteTaker")
    
    chart_generating_agent = create_react_agent(llm, tools=[read_document, python_repl])
    context_aware_chart_generating_agent = prelude | chart_generating_agent
    chart_generating_node = functools.partial(agent_node, agent=context_aware_chart_generating_agent, name="ChartGenerator")
    
    doc_writing_supervisor = create_team_supervisor(
        llm,
        "You are a supervisor tasked with managing a conversation between the"
        " following workers:  {team_members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH.",
        ["DocWriter", "NoteTaker", "ChartGenerator"],
    )

    authoring_graph = StateGraph(DocWritingState)
    authoring_graph.add_node("DocWriter", doc_writing_node)
    authoring_graph.add_node("NoteTaker", note_taking_node)
    authoring_graph.add_node("ChartGenerator", chart_generating_node)
    authoring_graph.add_node("supervisor", doc_writing_supervisor)

    authoring_graph.add_edge("DocWriter", "supervisor")
    authoring_graph.add_edge("NoteTaker", "supervisor")
    authoring_graph.add_edge("ChartGenerator", "supervisor")
    authoring_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "DocWriter": "DocWriter",
            "NoteTaker": "NoteTaker",
            "ChartGenerator": "ChartGenerator",
            "FINISH": END,
        },
    )
    authoring_graph.add_edge(START, "supervisor")
    
    return authoring_graph.compile()

def create_super_graph():
    """Create the top-level supervisor graph."""
    class State(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
        next: str

    llm = ChatOpenAI(model="gpt-4", temperature=0)

    research_chain = create_research_team()
    authoring_chain = create_doc_writing_team()

    supervisor_node = create_team_supervisor(
        llm,
        "You are a supervisor tasked with managing a conversation between the"
        " following teams: {team_members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH.",
        ["ResearchTeam", "PaperWritingTeam"],
    )
    
    def get_last_message(state: State) -> dict:
        """Get the last message from state and return it as a dictionary."""
        return {"messages": [state["messages"][-1]]}
    
    def join_graph(response: dict) -> dict:
        """Join the graph response with the current state."""
        return {"messages": [response["messages"][-1]]}

    super_graph = StateGraph(State)
    super_graph.add_node("ResearchTeam", get_last_message | research_chain | join_graph)
    super_graph.add_node("PaperWritingTeam", get_last_message | authoring_chain | join_graph)
    super_graph.add_node("supervisor", supervisor_node)

    super_graph.add_edge("ResearchTeam", "supervisor")
    super_graph.add_edge("PaperWritingTeam", "supervisor")
    super_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "PaperWritingTeam": "PaperWritingTeam",
            "ResearchTeam": "ResearchTeam",
            "FINISH": END,
        },
    )
    super_graph.add_edge(START, "supervisor")
    
    return super_graph.compile()

def main():
    """Run the hierarchical agent system with example queries."""
    super_graph = create_super_graph()

    print("Example: Research Report on North American Sturgeon")
    initial_state = {
        "messages": [
            HumanMessage(
                content="Write a brief research report on the North American sturgeon. Include a chart."
            )
        ],
        "next": "ResearchTeam"
    }
    
    for s in super_graph.stream(
        initial_state,
        {"recursion_limit": 150},
    ):
        if "__end__" not in s:
            print(s)
            print("----")

if __name__ == "__main__":
    main() 