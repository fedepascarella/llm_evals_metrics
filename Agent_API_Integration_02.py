import requests
import json

# Define the URL
url = 'https://cap-kairo-aa-dev.niceground-82e6c647.eastus.azurecontainerapps.io/api/chat/agent/'

# Example query
query = "What is the relationship between fermentative and electroactive bacteria in biofilms, as discussed in Data Chunk 2?"

# Function to send a query to the API and get the response


def get_response(query):
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": query
            },
        ],
        "common": {
            "systemPrompt": "You are a helpful assistant.",
            "llmModelType": "gpt",
            "llmModelProvider": "azure-openai",
            "llmModelVersion": "gpt-4",
            "previousMessages": 10,
            "temperature": 0.7,
            "maxTokens": 2048,
            "topP": 0.95,
            "frequencyPenalty": 0,
            "presencePenalty": 0,
            "stop": []
        },
        "type": "retrieval-agent",
        "persona": {
            "retriever": {
                "collection": "test",
                "top_k": 5,
                "search_engine": "lc_azure_search",
                "similarity_function": "cosine",
                "strategy": "similarity",
                "embedding_model": "sentence-transformers/all-mpnet-base-v2",
                "context_relevance_threshold": 0.2
            },
            "rag": {
                "prompt": "Answer the following question based only on the provided context:\n\n<context>\n{context}\n</context>\n\nQuestion: {question}."
            }
        }
    }

    response = requests.post(url, data=json.dumps(payload), headers={
                             'Content-Type': 'application/json'}, stream=True)

    if response.status_code == 200:
        search_results = []
        response_content = []
        for line in response.iter_lines():
            if line:
                try:
                    response_content.append(line.decode('utf-8'))
                    json_line = json.loads(line)
                    if "retriever_response" in json_line.get("metadata", {}):
                        search_results = json_line["metadata"]["retriever_response"]["search_results"]
                        break  # No need to continue once we've found the retriever_response
                except json.JSONDecodeError:
                    print(f"Failed to decode line: {line}")

        # Print the entire response
        print("Full response:")
        for part in response_content:
            print(part)

        return search_results
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


# Get the response for the example query
search_results = get_response(query)

# Check and print the content_page values if there is data
if search_results:
    for result in search_results:
        if 'content_page' in result:
            print(result['content_page'])
else:
    print("No data found in search_results.")
 