##from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
##from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms import LlamaCpp
llm = LlamaCpp(model_path="./models/dolphin-2.2.1-mistral-7b.Q2_K.gguf",temperature=0.0)


tools = load_tools(["llm-math","wikipedia"], llm=llm)

agent= initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)

agent("What is the 25% of 300?")