import os
import json
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

date = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
config_Test_Case_Agent = input_data['TEST_CASE_AGENT']
quantity_Test_Cases = config_Test_Case_Agent['test_cases_quantity']
test_types = config_Test_Case_Agent['testing_type']

TEST_CASE_AGENT_SYSTEM_PROMPT = f"""
Always follow this instructions. You are an LLM agent designed to generate this amount {quantity_Test_Cases} of QA test cases. Follow these steps:

1. For each test case, generate a detailed test case, including:
   - Test Case ID
   - Date
   - Description
   - Preconditions
   - Steps
   - Expected Results

Present the {quantity_Test_Cases} output(s) in JSON format following this structure:
    {{
    "test_cases": [
        {{
        "test_case_id": "TC001",
        "creation_date": "{date}",
        "description": "Verify user login functionality",
        "preconditions": ["User is registered", "User is on the login page"],
        "steps": ["Enter valid username", "Enter valid password", "Click on the login button"],
        "expected_results": ["User is redirected to the dashboard"]
        }},
        {{
        "test_case_id": "TC002",
        "creation_date": "{date}",
        "description": "Verify user logout functionality",
        "preconditions": ["User is logged in"],
        "steps": ["Click on the logout button"],
        "expected_results": ["User is redirected to the login page"]
        }}
    ]
    }}

"""

enabled_tests_prompt = "Generate test cases for the following types of testing:"

if test_types['functional']:
    enabled_tests_prompt += """
Functional Testing:
- Verify that the LLM app functions as intended and meets specified requirements.
- Test cases should cover a variety of valid and invalid inputs, feature coverage, and response accuracy.
"""

if test_types['performance']:
    enabled_tests_prompt += """
Performance Testing:
- Assess the LLM app’s performance under various conditions, including response time, scalability, and resource usage.
- Create test cases that simulate high load and stress conditions to evaluate performance.
"""

if test_types['security']:
    enabled_tests_prompt += """
Security Testing:
- Identify vulnerabilities in the LLM app and ensure protection against threats and attacks.
- Test input prompt injection, personal information, and data privacy.
"""

if test_types['usability']:
    enabled_tests_prompt += """
Usability Testing:
- Ensure the LLM app is user-friendly and provides a good user experience.
- Test the usability of the interface, interaction flow, and error handling.
"""

if test_types['bias_fairness']:
    enabled_tests_prompt += """
Bias and Fairness Testing:
- Ensure that the LLM app is free from bias and treats all inputs fairly.
- Create test cases to detect any biases, ensure fairness and inclusivity, and adhere to ethical guidelines.
"""

# Combine the base prompt with enabled tests instructions
full_prompt = TEST_CASE_AGENT_SYSTEM_PROMPT + enabled_tests_prompt

print(full_prompt)

# Initialize the AzureOpenAI LLM
llm = AzureChatOpenAI(deployment_name="gpt4-o", verbose=True,
                      temperature=0.5)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=TEST_CASE_AGENT_SYSTEM_PROMPT),
    HumanMessage(content="{full_prompt}")
])

output_parser = StrOutputParser()

# Create and run the chain
chain = prompt | llm | output_parser

result = chain.invoke({"input": full_prompt})
print(result)

output_data = {
    "test_cases": [
        {
            "test_case_id": "TC001",
            "creation_date": "10-05-2023 14:30:00",
            "description": "Verify user login functionality",
            "preconditions": ["User is registered", "User is on the login page"],
            "steps": ["Enter valid username", "Enter valid password", "Click on the login button"],
            "expected_results": ["User is redirected to the dashboard"]
        },
        {
            "test_case_id": "TC002",
            "creation_date": "10-05-2023 14:35:00",
            "description": "Verify user logout functionality",
            "preconditions": ["User is logged in"],
            "steps": ["Click on the logout button"],
            "expected_results": ["User is redirected to the login page"]
        },
        {
            "test_case_id": "TC003",
            "creation_date": "10-05-2023 14:40:00",
            "description": "Verify password reset functionality",
            "preconditions": ["User is on the login page"],
            "steps": ["Click on the 'Forgot Password' link", "Enter registered email address", "Click on the 'Reset Password' button"],
            "expected_results": ["User receives a password reset email"]
        },
        {
            "test_case_id": "TC004",
            "creation_date": "10-05-2023 14:45:00",
            "description": "Verify user registration functionality",
            "preconditions": ["User is on the registration page"],
            "steps": ["Enter valid username", "Enter valid email", "Enter valid password", "Click on the 'Register' button"],
            "expected_results": ["User receives a confirmation email", "User is redirected to the login page"]
        }
    ]
}

# Load existing results from JSON file if it exists
if os.path.exists('test_cases_output_results.json'):
    with open('test_cases_output_results.json', 'r') as output_file:
        existing_data = json.load(output_file)
        if "results" not in existing_data:
            existing_data["results"] = []
else:
    existing_data = {"results": []}

# Append the new result to the existing data
existing_data["results"].append(output_data)

# Save the updated results back to the JSON file
with open('test_cases_output_results.json', 'w') as output_file:
    json.dump(existing_data, output_file, indent=4)
