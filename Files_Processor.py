import pandas as pd

# Load the CSV files
api_results = pd.read_csv('api_results.csv')
questions_json_with_chunks = pd.read_csv('questions_json_with_chunks.csv')

# Create the new dataframe
new_df = pd.DataFrame()

# Mapping columns from api_results.csv
new_df['query'] = api_results['Query']
new_df['actual_output'] = api_results['Response']
new_df['retrieval_context'] = api_results['Search_Results']

# Mapping columns from questions_json_with_chunks.csv
new_df['expected_output'] = questions_json_with_chunks['Data Chunks Text']
new_df['context'] = questions_json_with_chunks['Data Chunks Text']

# Export the new dataframe to a CSV file
new_df.to_csv('Eval_dataframe.csv', index=False)

print("New dataframe has been created and saved as 'new_dataframe.csv'")
