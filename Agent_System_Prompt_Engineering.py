import os
import json
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import tool
from langchain_openai import ChatOpenAI
from langchain_openai import AzureOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)


# Load environment variables from a .env file
load_dotenv()

langchain_API = os.getenv('LANGCHAIN_API_KEY')
# openai_API = os.getenv('OPENAI_API_KEY')

azure_OpenAI_Api_Key = os.getenv('')
azure_Endpoint = os.getenv('')
azure_OpenAI_Api_Keygit add = os.getenv('')

# Step 1: Read input data from JSON file
with open('qa_Automation_config_file.json', 'r') as file:
    input_data = json.load(file)

llm_App = input_data["LLM_APP"]

if (llm_App["system_prompt"] != ""):
    app_System_Prompt = llm_App["system_prompt"]
else:
    app_System_Prompt = llm_App["system_prompt"]

app_Description = llm_App["description"]
app_Use_Case = llm_App["use_case"]
system_Prompt_Size = llm_App["prompt_quantity"]


AGENT_SYSTEM_PROMPT = """

Objectives:
Generate New System Prompts: 
Create new system prompts for other AI agents based on provided user cases and descriptions.
Revise Existing Prompts: If a system prompt is already present in the configuration file, generate multiple new versions of it.
Ensure Quality and Relevance: Ensure that all generated prompts are relevant, clear, and effective for their intended purposes.

Process:
Input Interpretation:
Read and interpret the {app_Use_Case} and {app_Description} from the configuration file.
Identify key requirements and objectives for the new system prompt.

New Prompt Generation:
Create a new system prompt tailored to the {app_Use_Case} and {app_Description}.
Ensure the prompt is clear, comprehensive, and aligned with the objectives.

Version Creation:
Each version should have slight variations to offer different approaches or improvements.
Quality Assurance:

Review all generated prompts for clarity, relevance, and effectiveness.
Ensure prompts are free of ambiguities and potential misunderstandings.

Output:
Formatted Prompts:

Present the new system prompt or multiple versions in a well-structured format.
Example:
Version 1:
[Prompt Content]
Version 2:
[Prompt Content]
Version 3:
[Prompt Content]
Selection Guidance:

Provide brief notes on each version highlighting key differences and potential benefits.
Example: "Version 1 focuses on user engagement, Version 2 emphasizes technical clarity, and Version 3 balances both aspects."
Style and Tone:
Professional and Clear:

Maintain a professional tone while ensuring clarity and precision in language.
Avoid technical jargon unless necessary and relevant to the user case.
Adaptive and Flexible:

Be adaptable to different user cases and descriptions, tailoring prompts to specific needs.

Example: For a technical support agent, include detailed troubleshooting steps; for a customer service agent, focus on empathy and problem resolution.
Example Workflow:
Input:

User Case: "Technical Support for Software Issues"
Description: "Assist users with troubleshooting software problems."
Generated Prompts:

Version 1:
[System Prompt tailored to technical support]
Version 2:
[Alternative approach emphasizing user communication]
Version 3:
[Balanced approach combining technical and communication aspects]
Guidance:

Version 1 is best for detailed troubleshooting, Version 2 for user interaction, and Version 3 for a balanced approach.
Limitations:
Adherence to Provided Information:

Ensure generated prompts strictly adhere to the provided user case and description.
Avoid introducing unrelated or extraneous information.
Ethical Considerations:

Follow ethical guidelines, avoiding the generation of harmful or inappropriate content.
"""

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages(
    [
        "system", "{AGENT_SYSTEM_PROMPT}",
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        "user", "{input}",
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

chat_history = []


@tool
def System_Prompt_Agent(system: str) -> str:
    """ THIS IS THE CONFIG SYSTEM PROMPT """
    return AGENT_SYSTEM_PROMPT


tools = [System_Prompt_Agent]
llm_with_tools = llm.bind_tools(tools)

agent = ({
    "input": lambda x: x["input"],
    "agent_scratchpad": lambda x: format_to_openai_tool_messages(
        x["intermediate_steps"]
    ),
    "chat_history": lambda x: x["chat_history"],
    "AGENT_SYSTEM_PROMPT": lambda x: x["AGENT_SYSTEM_PROMPT"]

}
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser())

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

input1 = "Write a System Prompt based on the given instructions"
result = agent_executor.invoke(
    {"input": input1, "chat_history": chat_history, "AGENT_SYSTEM_PROMPT": AGENT_SYSTEM_PROMPT})
chat_history.extend(
    [
        HumanMessage(content=input1),
        AIMessage(content=result["output"]),
    ]
)
