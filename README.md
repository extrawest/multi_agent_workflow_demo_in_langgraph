# 🤖 Multi-Agent Workflow Demonstrations in LangGraph

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]()
[![Maintainer](https://img.shields.io/static/v1?label=Yevhen%20Ruban&message=Maintainer&color=red)](mailto:yevhen.ruban@extrawest.com)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub release](https://img.shields.io/badge/release-v1.0.0-blue)

A collection of demonstrations showcasing different patterns for implementing multi-agent workflows using LangGraph. 
Each example highlights specific orchestration approaches to help developers understand and build collaborative AI systems.

## 📋 Features

### Hierarchical Agent Teams
- 🏗️ Orchestrates complex workflows with multiple levels of supervision
- 🔄 Seamlessly coordinates specialized agent teams (research, document writing)
- 🧠 Efficiently delegates tasks to lower-level agents with appropriate tools
- 📊 Follows proper escalation and reporting paths in the agent hierarchy

### Agent Supervision
- 👨‍💼 Models a supervisor-worker relationship for intelligent task delegation
- 🔄 Routes tasks between research and coding agents based on requirements
- 🚦 Manages conversation flow with clear transitions between agents
- 🔍 Makes real-time decisions about which agent should act next

### Multi-Agent Collaboration
- 🤝 Enables direct peer-to-peer collaboration between agents
- 🛠️ Shares tools and information across collaborating agents
- 📊 Demonstrates fluid conversation flow between specialized agents
- 📡 Facilitates tool calling between different agent types

## 🏗️ Architecture

The demonstrations showcase three distinct multi-agent architecture patterns:

Hierarchical Teams

![hierarchical_agent_teams](https://github.com/user-attachments/assets/0be43be6-a083-46d8-8f6e-024ee5c9b049)

Agent Supervisor

![agent_supervisor](https://github.com/user-attachments/assets/46ef788f-d529-4903-a8c0-e10275e58826)

Multi-Agent Collaboration

![multi_agent_collaboration](https://github.com/user-attachments/assets/d8714163-8836-4365-9436-cce0755ef9dc)


## 📦 Implementation Details

### [`hierarchical_agent_teams.py`](./scripts/hierarchical_agent_teams.py)
This demonstration implements a sophisticated hierarchical team structure with multiple levels of supervision:

- **Top-level Supervisor**: Coordinates between specialized teams
- **Research Team**: Combines web search and web scraping capabilities 
- **Document Writing Team**: Creates, edits and manages document creation
- **Tool Integration**: Implements document creation/editing tools, search tools, and web scraping
- **State Management**: Shows how to manage complex state across the hierarchy

### [`agent_supervisor.py`](./scripts/agent_supervisor.py)
This demonstration implements a supervisor-worker architecture:

- **Supervisor Agent**: Makes routing decisions about which agent to activate
- **Research Agent**: Uses search tools to gather information
- **Coding Agent**: Executes Python code for calculations and analysis
- **Decision Logic**: Shows how to implement routing logic for multi-agent systems

### [`multi_agent_collaboration.py`](./scripts/multi_agent_collaboration.py)
This demonstration implements a peer-to-peer collaborative agent system:

- **Researcher Agent**: Gathers data from web sources
- **Chart Generator Agent**: Creates visualizations from research data
- **Tool Sharing**: Shows how tools can be used across agent boundaries
- **Collaborative Workflow**: Demonstrates agents working together on a shared task

## 🛠️ Requirements

- Python 3.9+
- OpenAI API key
- Tavily API key 
- LangChain and LangGraph libraries

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-openai-api-key"
export TAVILY_API_KEY="your-tavily-api-key"

# Optional: Enable LangChain tracing
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_PROJECT="Multi-Agent-Workflow"
```

## 🚀 Usage

Each demonstration can be run as a standalone script:

```python
# Run the hierarchical agent teams demo
python scripts/hierarchical_agent_teams.py

# Run the agent supervisor demo
python scripts/agent_supervisor.py

# Run the multi-agent collaboration demo
python scripts/multi_agent_collaboration.py
```

## 🏆 Key Benefits

- **Modularity**: Each example demonstrates a reusable pattern for multi-agent systems
- **Flexibility**: Shows different approaches to agent coordination and collaboration
- **Adaptability**: Provides templates that can be customized for specific use cases
- **Scalability**: Demonstrates patterns that can scale to complex multi-agent systems

Developed by [extrawest](https://extrawest.com/). Software development company
