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
from langchain_core.output_parsers import JsonOutputParser
from abc import ABC, abstractmethod
from typing import List
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from docx import Document as DocxDocument  # Import for reading .docx files

# Load environment variables from a .env file
load_dotenv()

# Fetch environment variables
azure_OpenAI_Api_Key = os.getenv('AZURE_OPENAI_API_KEY')
azure_Endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
azure_OpenAI_Version = os.getenv('OPENAI_API_VERSION')

# Function to get the number of pages in a PDF
def get_num_pages(pdf_path):
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            return len(reader.pages)
    except (PdfReadError, FileNotFoundError) as e:
        print(f"Error reading {pdf_path}: {e}")
        return 0

# Function to randomly select a page number from a PDF or DOCX
def select_random_page(num_pages):
    return random.randint(1, num_pages)

# Function to get the number of paragraphs (treated as "pages") in a DOCX file
def get_num_paragraphs(docx_path):
    try:
        doc = DocxDocument(docx_path)
        return len(doc.paragraphs)
    except Exception as e:
        print(f"Error reading {docx_path}: {e}")
        return 0

# Function to read a specific paragraph ("page") from a DOCX file
def read_docx_page(docx_path, page_num):
    try:
        doc = DocxDocument(docx_path)
        return doc.paragraphs[page_num - 1].text
    except Exception as e:
        print(f"Error reading page {page_num} of {docx_path}: {e}")
        return ""

# Class to represent a document
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

# Load PDFs and DOCX files and get random pages
file_path = "Data/PDFs/"  # Set your directory path here
pdf_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.pdf')]
docx_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.docx')]

# List to hold Document objects
docs = []

# Dictionary to store the number of pages/paragraphs and random page/paragraph number for each document
doc_info = {}

# Load PDFs and get page information
for pdf_file in pdf_files:
    num_pages = get_num_pages(pdf_file)
    if num_pages > 0:
        random_page = select_random_page(num_pages)
        doc_info[os.path.basename(pdf_file)] = {
            'total_pages': num_pages,
            'random_page': random_page
        }
        # Add Document objects to the list
        reader = PdfReader(pdf_file)
        for i in range(num_pages):
            page_content = reader.pages[i].extract_text()
            docs.append(Document(page_content, {'source': os.path.basename(pdf_file), 'page': i + 1}))

# Load DOCX files and get paragraph information (treated as pages)
for docx_file in docx_files:
    num_paragraphs = get_num_paragraphs(docx_file)
    if num_paragraphs > 0:
        random_page = select_random_page(num_paragraphs)
        doc_info[os.path.basename(docx_file)] = {
            'total_paragraphs': num_paragraphs,
            'random_page': random_page  # Use the same key for consistency
        }
        # Add Document objects to the list
        for i in range(num_paragraphs):
            paragraph_content = read_docx_page(docx_file, i + 1)
            if paragraph_content.strip():  # Avoid adding empty paragraphs
                docs.append(Document(paragraph_content, {'source': os.path.basename(docx_file), 'page': i + 1}))

# Check if documents were loaded
if not docs:
    raise ValueError("No documents were loaded")

# Function to export page content to a text file
def export_page_content_to_txt(docs, output_file, doc_info):
    with open(output_file, 'w') as file:
        for doc in docs:
            source = doc.metadata['source']
            page = doc.metadata['page']
            if source in doc_info and doc_info[source].get('random_page', 0) == page:
                file.write(f"Source: {source}, Page: {page}\n")
                file.write(doc.page_content + "\n\n")

# Export content of randomly selected pages to a text file
output_file = 'random_page_contents.txt'
export_page_content_to_txt(docs, output_file, doc_info)
print(f"Randomly selected page content has been exported to {output_file}")

# Class to split text into chunks
class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size=900, chunk_overlap=0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks

# Create chunks based on the randomly selected page
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
data = []

for doc in docs:
    source = doc.metadata['source']
    page = doc.metadata['page']

    # Debugging: Print current document info
    print(f"Processing document: {source}, Page: {page}")
    
    # Ensure that the document's source is in doc_info
    if source not in doc_info:
        print(f"Warning: {source} not found in doc_info. Skipping this document.")
        continue
    
    # Debugging: Print matching page info
    print(f"Expected random page for {source}: {doc_info[source].get('random_page', 0)}")
    
    # Check if the document matches the randomly selected page/paragraph
    if doc_info[source].get('random_page', 0) == page:
        print(f"Match found for {source} on Page {page}")
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            data.append({
                'page_content': chunk,
                'source': source,
                'page': page
            })
    else:
        print(f"No match for {source} on Page {page}")

# Check if data was added
if not data:
    print("No data was added. Please check the conditions and document processing.")
else:
    # Creating the DataFrame
    df = pd.DataFrame(data, columns=['page_content', 'source', 'page'])

    # Print the number of chunks obtained
    print(f"Number of chunks obtained: {len(data)}")
    print(df.head(100))

    # Access the first string safely
    if not df.empty:
        first_string = df['page_content'].iloc[0]
        print(f"First string: {first_string}")
    else:
        print("The DataFrame is empty. No content to display.")



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
filtered_df = filtered_df[filtered_df['Average Score'] > 3]

# Step 4: Create a new DataFrame with the filtered chunks
result_df = filtered_df[['chunk']]

# Enumerate the chunks
result_df.loc[:, 'Chunk Number'] = range(1, len(result_df) + 1)

# Display the result
print(result_df)

# Export the result to a CSV file
result_df.to_csv('Chunk_result.csv', index=False)


#####################################
###### DATASET DESIGNER AGENT #######
#####################################

# Load system prompts JSON file
with open('output_results.json', 'r') as file:
    system_prompts_data = json.load(file)

# Load test cases JSON file
with open('test_cases_output_results.json', 'r') as file:
    test_cases_data = json.load(file)

# Load the configuration file
with open('qa_Automation_config_file.json', 'r') as config_data_file:
    config_data = json.load(config_data_file)

llm_App = config_data["LLM_APP"]
app_type = llm_App["type"]
app_type_rag = app_type["rag"]
prompt_engineering = config_data["PROMP_ENGINEERING"]
dataset_number_questions = prompt_engineering["overall_Dataset_size"]
prompt_engineering_categories = prompt_engineering["categories"]
prompt_engineering_technique_zero_shot = prompt_engineering_categories["zero_shot"]

# Extract system prompts content
system_prompts = []
for result in system_prompts_data["results"]:
    if "versions" in result:
        for version in result["versions"]:
            system_prompts.append(version["content"])

# Extract test case descriptions
test_case_descriptions = []
for result in test_cases_data["results"]:
    if "test_cases" in result:
        for test_case in result["test_cases"]:
            test_case_descriptions.append(test_case["description"])

# Create the chain of thought system prompt

def create_chain_of_thought(system_prompts, test_case_descriptions, result_df, app_type_rag, dataset_number_questions, prompt_engineering_technique_zero_shot):
    cot_prompt = "You are an AI designed to generate a dataset of questions based on provided system prompts, test cases, and data chunks. Follow these steps:\n"

    cot_prompt += "1. Review the following system prompts:\n"
    for i, prompt in enumerate(system_prompts):
        cot_prompt += f"System Prompt {i+1}: {prompt}\n"

    cot_prompt += "\n2. Review the following test case descriptions:\n"
    for i, description in enumerate(test_case_descriptions):
        cot_prompt += f"Test Case {i+1}: {description}\n"

    if app_type_rag:
        cot_prompt += "\n3. Review the following data chunks:\n"
        if 'chunk' in result_df.columns:
            for i, row in result_df.iterrows():
                cot_prompt += f"Data Chunk {i+1}: {row['chunk']}\n"
        else:
            cot_prompt += "No data chunks available.\n"

    if prompt_engineering_technique_zero_shot:
        cot_prompt += "\n4. Apply zero-shot prompt engineering technique to generate questions. If data chunks are present, it is compulsory to generate questions based on them.\n"
    else:
        cot_prompt += "\n4. Based on the system prompts, test case descriptions, and data chunks (if any), generate a set of questions.\n"
    
    cot_prompt += f"Please generate a set of {dataset_number_questions} questions that could be used to test or clarify the system's functionality. Ensure the questions are relevant and cover various aspects of the described operations, test cases, and data chunks.\n"

    # Add instructions for JSON formatting
    cot_prompt += """
Please return the output in the following JSON format:

{
  "questions": [
    {
      "category": "Category Name",
      "question": "The question text",
      "data_chunks_used": ["Data Chunk X", "Data Chunk Y"],
      "data_chunks_text": ["Text of Data Chunk X", "Text of Data Chunk Y"],
      "explanation": "Explanation for this question"
    },
    ...
  ]
}
"""

    return cot_prompt


# Generate the chain of thought system prompt
cot_prompt = create_chain_of_thought(
    system_prompts, test_case_descriptions, result_df, app_type_rag, dataset_number_questions, prompt_engineering_technique_zero_shot)
print(cot_prompt)

dataset_generator_llm = AzureChatOpenAI(deployment_name="gpt4-o", verbose=True,
                                        temperature=0.8)

dataset_generator_llm_input = "Execute the system prompt"

dataset_system_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=cot_prompt),
    HumanMessage(content=dataset_generator_llm_input)
])

#json_output_parser = JsonOutputParser()

dataset_result_chain = dataset_system_prompt | dataset_generator_llm | output_parser
dataset_llm_result = dataset_result_chain.invoke(
    {"input": dataset_generator_llm_input})

print(f"DATASET: \n{dataset_llm_result}")

# Assuming `llm_response` contains the JSON string returned by the LLM
llm_response = dataset_llm_result

# Parse the JSON response
response_data = json.loads(llm_response)

# Extract the questions into a DataFrame
questions_data = []
for item in response_data["questions"]:
    questions_data.append({
        "Category": item["category"],
        "Question": item["question"],
        "Data Chunks Used": ", ".join(item["data_chunks_used"]),  # Join the list of data chunks into a single string
        "Data Chunks Text": " | ".join(item["data_chunks_text"]),  # Join the chunk texts with a separator for clarity
        "Explanation": item["explanation"]
    })

# Convert to DataFrame
questions_df = pd.DataFrame(questions_data)

# Display the DataFrame
print("Questions DataFrame:")
print(questions_df)

# Export the DataFrame to a CSV file
questions_df.to_csv('questions_json_with_chunks.csv', index=False)
print("DataFrame has been exported to 'questions_json_with_chunks.csv'.")