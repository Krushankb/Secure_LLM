from llama_cpp import Llama
from langchain.agents.agent_types import AgentType
 
llamallm = Llama(model_path="./models/llama-2-7b-chat.ggmlv3.q4_1.bin",n_ctx=2048)
 
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from langchain import PromptTemplate, LLMChain
import requests
class CustomLLM(LLM):
  def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
    print("***\n"+prompt+"\n***")
    output = llamallm(prompt, echo=False)
    output = output["choices"][0]["text"]
    if(output.find("\nAction:")>=0 and output.find("\nObservation:")>output.find("\nAction:")): return(output[0:output.find("\nObservation:")])
    else: return(output)
  @property
  def _llm_type(self) -> str:
    return "custom"
 
llm=CustomLLM()
 
from langchain import LLMMathChain
from langchain.agents import AgentType, initialize_agent
 
llm_math_chain = LLMMathChain(llm=llm, verbose=True)
from pydantic import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, Tool, tool
 
class CalculatorInput(BaseModel):
    question: str = Field()
 
tools = [
    Tool.from_function(
        func=llm_math_chain.run,
        name="Calculator",
        description="useful for when you need to answer questions about math",
        args_schema=CalculatorInput
    )
]
 
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, stop=["\nThought:"]
)
agent.run("What is 3 to the power of 2?")



# llm_test.py continuation:

# class DBInput(BaseModel):
#   question: str = Field()

# # llm_math_chain = LLMMathChain(llm=llm, verbose=True)
# # class CalculatorInput(BaseModel):
# #   question: str = Field()
 
# tools = [
#     # Tool.from_function(
#     #     func=llm_math_chain.run,
#     #     name="Calculator",
#     #     description="useful for when you need to answer questions about math",
#     #     args_schema=CalculatorInput
#     # ),
#     Tool.from_function(
#         func=db.run,
#         name="Database",
#         description="Useful for executing sql queries",
#         args_schema=DBInput
#     )
# ]

# # Set up the base template
# template = """Answer the following question as best you can. You have access to the following tools:

# {tools}

# Use the following format:

# Question: the input question you must answer
# Action: choose Database
# Action Input: the input to the action
# Observation: the result of the action
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question

# Begin!

# Question: {input}"""
# # ... (this Thought/Action/Action Input/Observation can repeat N times)
# # {agent_scratchpad}"""
# # https://github.com/langchain-ai/langchain/issues/2980

# # Set up a prompt template
# class CustomPromptTemplate(StringPromptTemplate):
#     # The template to use
#     template: str
#     # The list of tools available
#     tools: List[Tool]

#     def format(self, **kwargs) -> str:
#         # Get the intermediate steps (AgentAction, Observation tuples)
#         # Format them in a particular way
#         intermediate_steps = kwargs.pop("intermediate_steps")
#         thoughts = ""
#         for action, observation in intermediate_steps:
#             thoughts += action.log
#             thoughts += f"\nObservation: {observation}\nThought: "
#         # Set the agent_scratchpad variable to that value
#         # kwargs["agent_scratchpad"] = thoughts
#         # Create a tools variable from the list of tools provided
#         kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
#         # Create a list of tool names for the tools provided
#         kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
#         return self.template.format(**kwargs)
    
# prompt = CustomPromptTemplate(
#     template=template,
#     tools=tools,
#     # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
#     # This includes the `intermediate_steps` variable because that is needed
#     input_variables=["input", "intermediate_steps"]
# )

# class CustomOutputParser(AgentOutputParser):
#     def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
#         # Check if agent should finish
#         if "Final Answer:" in llm_output:
#             return AgentFinish(
#                 # Return values is generally always a dictionary with a single `output` key
#                 # It is not recommended to try anything else at the moment :)
#                 return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
#                 log=llm_output,
#             )
#         # Parse out the action and action input
#         regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
#         match = re.search(regex, llm_output, re.DOTALL)
#         if not match:
#             raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
#         action = match.group(1).strip()
#         action_input = match.group(2)
#         # Return the action and action input
#         return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
# output_parser = CustomOutputParser()

# # LLM chain consisting of the LLM and a prompt
# llm_chain = LLMChain(llm=llm, prompt=prompt)
# tool_names = [tool.name for tool in tools]
# agent = LLMSingleActionAgent(
#     llm_chain=llm_chain,
#     output_parser=output_parser,
#     stop=["\nQuestion:"],
#     allowed_tools=tool_names
# )

# agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True)
# text = "Execute the following SQL query with the Database tool and show the output: " + response + "."
# # text = "What is 3 to the power of 2?"
# print(text)
# agent_executor.run(text)