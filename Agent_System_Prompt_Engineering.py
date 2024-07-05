import os
import json
import re
from datetime import datetime
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
date = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

PROVIDED_SYSTEM_PROMPT = llm_App.get("given_system_prompt", "")

AGENT_SYSTEM_PROMPT = f"""
Objectives:
Generate New System Prompts: Create new system prompts for other AI agents based on provided user case: {app_Use_Case} and description: {app_Description}.
Create {system_Prompt_Size} new versions of it. Always start each prompt with the sentence "I am an agent that:"
Ensure Quality and Relevance: Ensure that all generated prompts are relevant, clear, and effective for their intended purposes.
Process:
Input Interpretation:
Read and interpret the use case: {app_Use_Case} and the description: {app_Description} from the configuration file.
Identify key requirements and objectives for the new system prompt.
New Prompt Generation:
Create {system_Prompt_Size} new system prompt(s) tailored to the {app_Use_Case} and {app_Description}.
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
Creation date: {date}
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

PROVIDED_AGENT_SYSTEM_PROMPT = f"""
Objectives:
Generate New System Prompts: Create new system prompts for other AI agents based on provided prompt {PROVIDED_SYSTEM_PROMPT}, user case: {app_Use_Case}, and description: {app_Description}.
Create {system_Prompt_Size} new versions of it.
Revise Existing Prompts: If a system prompt is already present in the configuration file, generate multiple new versions of it.
Ensure Quality and Relevance: Ensure that all generated prompts are relevant, clear, and effective for their intended purposes.
Process:
Input Interpretation:
Read and interpret the use case: {app_Use_Case} and the description: {app_Description} from the configuration file.
Identify key requirements and objectives for the new system prompt.
New Prompt Generation:
Create {system_Prompt_Size} new system prompt(s) tailored to the {app_Use_Case} and {app_Description}.
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
Creation date: {date}
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

input1 = "Create system prompts based on the instructions given."

# Initialize the AzureOpenAI LLM
llm = AzureChatOpenAI(deployment_name="gpt4-o", verbose=True,
                      temperature=0.5)


# Create the prompt template

if not PROVIDED_SYSTEM_PROMPT:

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content="{input}")
    ])
else:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=PROVIDED_AGENT_SYSTEM_PROMPT),
        HumanMessage(content="{input}")
    ])

output_parser = StrOutputParser()

# Create and run the chain
chain = prompt | llm | output_parser


result = chain.invoke({"input": input1})
print(result)

print("Raw result from LLM:", result)  # Debug print

# Parse the raw result into a structured format


def parse_llm_output(output):
    pattern = r"Version \d+:\s*(.*?)\s*(?=\n###|Version \d+|Selection Guidance:|$)"
    versions = re.findall(pattern, output, re.DOTALL)
    selection_guidance = re.search(
        r"Selection Guidance:\s*(.*)", output, re.DOTALL)

    if not versions:
        print("No versions found in the output.")
        return {"test_cases": []}

    parsed_data = {
        "creation_date": re.search(r"Creation date:\s*(.*)", output).group(1).strip(),
        "versions": [{"content": version.strip()} for version in versions]
    }

    if selection_guidance:
        parsed_data["selection_guidance"] = selection_guidance.group(
            1).strip().split("\n")

    return parsed_data


parsed_data = parse_llm_output(result)
print("Parsed output data:", parsed_data)  # Debug print

# Load existing results from JSON file if it exists
if os.path.exists('output_results.json'):
    with open('output_results.json', 'r') as output_file:
        existing_data = json.load(output_file)
        if "results" not in existing_data:
            existing_data["results"] = []
else:
    existing_data = {"results": []}

print("Existing data before appending new result:", existing_data)  # Debug print

# Append the new result to the existing data
existing_data["results"].append(parsed_data)

print("Data to be written to file:", existing_data)  # Debug print

# Save the updated results back to the JSON file
with open('output_results.json', 'w') as output_file:
    json.dump(existing_data, output_file, indent=4)
    print("Data successfully written to output_results.json")  # Debug print
