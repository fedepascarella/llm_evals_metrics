import os
import json
from dotenv import load_dotenv
from langchain.agents import tool
from langchain_openai import AzureChatOpenAI, AzureOpenAI
from langchain.schema import SystemMessage, HumanMessage, StrOutputParser
from langchain.prompts import ChatPromptTemplate


# Load environment variables from a .env file
load_dotenv()

# Fetch environment variablesq
azure_OpenAI_Api_Key = os.getenv('AZURE_OPENAI_API_KEY')
azure_Endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
azure_OpenAI_Version = os.getenv('OPENAI_API_VERSION')

# Step 1: Read input data from JSON file
with open('qa_Automation_config_file.json', 'r') as file:
    input_data = json.load(file)

llm_App = input_data["LLM_APP"]

app_Description = llm_App["description"]
app_Use_Case = llm_App["use_case"]
system_Prompt_Size = llm_App["prompt_quantity"]

AGENT_SYSTEM_PROMPT = """
Generate New System Prompts: 
Create new system prompts for other AI agents based on provided user cases here {use_case} and descriptions here {description}.
Revise Existing Prompts: If a system prompt is already present in the configuration file, generate multiple new versions of it.
Ensure Quality and Relevance: Ensure that all generated prompts are relevant, clear, and effective for their intended purposes.
"""

input1 = "Write a System Prompt based on the given instructions"

# Initialize the AzureOpenAI LLM
llm = AzureOpenAI(deployment_name="Completion",
                  temperature=0.5)

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=AGENT_SYSTEM_PROMPT),
    HumanMessage(content="{input}")
])

output_parser = StrOutputParser()

# Create and run the chain
chain = prompt | llm | output_parser


result = chain.invoke({"input": input1})
print(result)
