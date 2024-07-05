import os
import random
import pandas as pd
import json
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAI
from langchain.schema import SystemMessage, HumanMessage, StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from abc import ABC, abstractmethod
from typing import List
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

# Load environment variables from a .env file
load_dotenv()

# Fetch environment variables
azure_OpenAI_Api_Key = os.getenv('AZURE_OPENAI_API_KEY')
azure_Endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
azure_OpenAI_Version = os.getenv('OPENAI_API_VERSION')


##############################
###### DOCUMENTS PAGE COUNT ##
##############################

# Function to get the number of pages in a PDF
def get_num_pages(pdf_path):
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            return len(reader.pages)
    except (PdfReadError, FileNotFoundError) as e:
        print(f"Error reading {pdf_path}: {e}")
        return 0


# Load PDFs
file_path = "Data/PDFs/"  # Set your directory path here
pdf_files = [os.path.join(file_path, f)
             for f in os.listdir(file_path) if f.endswith('.pdf')]
pages_docs = [file for file in pdf_files if get_num_pages(
    file) > 0]  # Only keep valid PDFs

# Check if documents were loaded
if not pages_docs:
    raise ValueError("No documents were loaded")

# Get the total number of pages for each document
pdf_pages = {os.path.basename(file): get_num_pages(file) for file in pages_docs}

# Randomly select a page number for each document
random_pages = {pdf: random.randint(1, pages)
                for pdf, pages in pdf_pages.items()}

# Print the total number of pages and the randomly selected page for each document
print("Total pages and randomly selected page for each document:")
for pdf, pages in pdf_pages.items():
    print(f"{pdf}: {pages} pages, Random Page: {random_pages[pdf]}")


##############################
###### TEXT EXTRACTION #######
##############################

# List files in the directory
files = os.listdir(file_path)
print("Files in directory:", files)

# # Check if there are PDF files in the directory
# pdf_files = [f for f in files if f.endswith('.pdf')]
# if not pdf_files:
#     raise FileNotFoundError("No PDF files found in the directory")

# Load PDFs
loader = PyPDFDirectoryLoader(file_path)
docs = loader.load()

# Check if documents were loaded
if not docs:
    raise ValueError("No documents were loaded")

# Print first document
# print(docs[1])

# Assuming docs is a list of Document objects with the specified structure


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


# Function to export page_content of random pages to a text file
def export_page_1_content_to_txt(docs, output_file):
    with open(output_file, 'w') as file:
        for doc in docs:
            if doc.metadata['page'] == 1:
                file.write(doc.page_content + "\n\n")


# Specify the output file path
output_file = 'page_1_contents.txt'

# Call the function to export page_content of page 1
export_page_1_content_to_txt(docs, output_file)

print(f"Page 1 content has been exported to {output_file}")

data = []
text_chunks = []

# Split the extracted data into text chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900, chunk_overlap=0)


# create chunks based on the randomly selected page
for doc in docs:
    if doc.metadata['page'] == 1:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            data.append(
                {'page_content': chunk, 'source': doc.metadata['source'], 'page': doc.metadata['page']})

# Print the number of chunks obtained
print(len(data))
print(data)

# Creating the DataFrame
df = pd.DataFrame(data, columns=['page_content', 'source', 'page'])

print(df.head(100))


#####################################
###### CHUNKS EVALUATING AGENT ######
#####################################

EVALUATING_CHUNKS_AGENT_SYSTEM_PROMPT = f"""
    You are an evaluation agent tasked with assessing the quality of text chunks for question generation. Your goal is to determine if each chunk contains sufficient logical content to create meaningful and coherent questions. Follow these guidelines to evaluate each chunk:

Content Relevance:

Ensure the chunk contains relevant information that can be the basis of a question.
The information should be specific enough to allow for a focused question.
Clarity:

The text within the chunk should be clear and comprehensible.
Avoid chunks with ambiguous or vague content.
Logical Coherence:

The chunk should present information in a logically coherent manner.
Identify if the chunk has a clear main idea or point.
Question Potential:

Evaluate if the chunk contains potential for generating specific, detailed questions.
Determine if the information can lead to questions of varying complexity (e.g., factual, analytical).
Completeness:

The chunk should be sufficiently complete to form a standalone question without needing excessive additional context.
Avoid overly fragmented chunks that lack a coherent thought.
For each text chunk, provide a rating on a scale of 1 to 5 for the following criteria:

Content Relevance
Clarity
Logical Coherence
Question Potential
Completeness
Additionally, provide a brief justification for your rating, highlighting key points that influenced your evaluation.
"""


# Extract the chunks from the DataFrame
dataframe_chunks = df['page_content'].tolist()
# print(dataframe_chunks)

first_string = df['page_content'].iloc[0]
# print(f"First chunk for verification: {first_string}")

# Initialize the AzureOpenAI LLM
chunk_evaluating_llm = AzureChatOpenAI(deployment_name="gpt4-o", verbose=True,
                                       temperature=0)

output_parser = StrOutputParser()


def extract_scores(evaluation):
    scores = {
        "Content Relevance": 0,
        "Clarity": 0,
        "Logical Coherence": 0,
        "Question Potential": 0,
        "Completeness": 0
    }
    for line in evaluation.split("\n"):
        for category in scores.keys():
            if category in line:
                try:
                    score = int(line.split(":")[1].strip().replace('*', ''))
                    scores[category] = score
                except ValueError:
                    continue
    return scores


# Create and run the chain for each chunk
results = []
for dataframe_chunk in dataframe_chunks:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=EVALUATING_CHUNKS_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=dataframe_chunk)
    ])
    # input_data = {"dataframe_chunk": dataframe_chunk}
    # Debug statement to print the chunk being processed
    print(f"Processing chunk: {dataframe_chunk}\n")
    result = prompt | chunk_evaluating_llm | output_parser
    result = result.invoke({"input": dataframe_chunk})
    # results.append(result)
    scores = extract_scores(result)
    results.append({"chunk": dataframe_chunk,
                   "evaluation": result, "Scores": scores})


# Print the results
for i, result in enumerate(results):
    print(f"Chunk {i+1} Evaluation:")
    print(result)
    print("\n")

data = []
for result in results:
    chunk = result['chunk']
    scores = result['Scores']
    data.append({
        'chunk': chunk,
        'Content Relevance': scores['Content Relevance'],
        'Clarity': scores['Clarity'],
        'Logical Coherence': scores['Logical Coherence'],
        'Question Potential': scores['Question Potential'],
        'Completeness': scores['Completeness']
    })

# Create DataFrame
df = pd.DataFrame(data, columns=['chunk', 'Content Relevance', 'Clarity',
                  'Logical Coherence', 'Question Potential', 'Completeness'])

print(df)

# Step 1: Filter the DataFrame
filtered_df = df[['chunk', 'Content Relevance', 'Question Potential']].copy()

# Step 2: Calculate the average of 'Content Relevance' and 'Question Potential'
filtered_df['Average Score'] = filtered_df[[
    'Content Relevance', 'Question Potential']].mean(axis=1)

# Step 3: Filter the DataFrame to keep rows where the average score is above 4
filtered_df = filtered_df[filtered_df['Average Score'] > 4]

# Step 4: Create a new DataFrame with the filtered chunks
result_df = filtered_df[['chunk']]

# Display the result
print(result_df)


#####################################
###### DATASET DESIGNER AGENT #######
#####################################

# # Load system prompts JSON file
# with open('output_results.json', 'r') as file:
#     system_prompts_data = json.load(file)

# # Load test cases JSON file
# with open('test_cases_output_results.json', 'r') as file:
#     test_cases_data = json.load(file)

# # Extract system prompts content
# system_prompts = []
# for result in system_prompts_data["results"]:
#     if "versions" in result:
#         for version in result["versions"]:
#             system_prompts.append(version["content"])

# # Extract test case descriptions
# test_case_descriptions = []
# for result in test_cases_data["results"]:
#     if "test_cases" in result:
#         for test_case in result["test_cases"]:
#             test_case_descriptions.append(test_case["description"])

# # Create the chain of thought system prompt


# def create_chain_of_thought(system_prompts, test_case_descriptions):
#     cot_prompt = "You are an AI designed to generate a dataset of questions based on system prompts and test cases. Follow these steps:\n"

#     cot_prompt += "1. Review the following system prompts:\n"
#     for i, prompt in enumerate(system_prompts):
#         cot_prompt += f"System Prompt {i+1}: {prompt}\n"

#     cot_prompt += "\n2. Review the following test case descriptions:\n"
#     for i, description in enumerate(test_case_descriptions):
#         cot_prompt += f"Test Case {i+1}: {description}\n"

#     cot_prompt += "\n3. Based on the system prompts and test case descriptions, generate a set of questions that could be used to test or clarify the system's functionality. Ensure the questions are relevant and cover various aspects of the described operations and test cases.\n"

#     cot_prompt += "4. Provide a detailed explanation of how each question is derived from the provided system prompts and test cases.\n"

#     cot_prompt += "\nExample Question Set:\n"
#     cot_prompt += "Question 1: What is the expected output when adding 15 and 10 using the calculator?\n"
#     cot_prompt += "Explanation: This question is derived from the test case description 'Verify the calculator's ability to correctly calculate the sum of two numbers' where the steps involve adding 15 and 10.\n"

#     cot_prompt += "\nNow, generate the questions and explanations based on the provided information."

#     return cot_prompt


# # Generate the chain of thought system prompt
# cot_system_prompt = create_chain_of_thought(
#     system_prompts, test_case_descriptions)
# print(cot_system_prompt)

# # Load the configuration file
# with open('qa_Automation_config_file.json', 'r') as config_data_file:
#     config_data = json.load(config_data_file)

# # Document extraction chunks (example chunks)
# document_extraction_approved_chunks = {}
# dataset_questions_amount = ""

# # Determine the app type and append relevant chunks
# app_types = config_data["LLM_APP"]["type"]


# # Print the updated configuration to verify
# print(json.dumps(config_data, indent=4))

# # Save the updated configuration if needed
# with open('updated_config.json', 'w') as file:
#     json.dump(config_data, file, indent=4)
