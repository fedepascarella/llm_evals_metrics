import os
import pandas as pd
import json
import ast
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.metrics import SummarizationMetric
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.metrics import ContextualRecallMetric
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics import HallucinationMetric
from deepeval.metrics import BiasMetric
from deepeval.metrics import ToxicityMetric
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset


# Load environment variables from a .env file
load_dotenv()

# Fetch environment variablesq
azure_OpenAI_Api_Key = os.getenv('AZURE_OPENAI_API_KEY')
azure_Endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
azure_OpenAI_Version = os.getenv('OPENAI_API_VERSION')


class AzureOpenAI(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "gpt4-o"


# Replace these with real values
custom_model = AzureChatOpenAI(
    openai_api_version=azure_OpenAI_Version,
    azure_deployment="gpt4-o",
    azure_endpoint=azure_Endpoint,
    openai_api_key=azure_OpenAI_Api_Key,
)
# Initialize the AzureOpenAI LLM
azure_openai = AzureOpenAI(model=custom_model)

# DATASET

dataset = EvaluationDataset()
dataset.add_test_cases_from_csv_file(
    # file_path is the absolute path to you .csv file
    file_path="QA Tests - Tasks # _ Kairo Retriever Agent - New Evals_02.csv",
    input_col_name="query",
    actual_output_col_name="actual_output",
    expected_output_col_name="expected_output",
    context_col_name="context",
    context_col_delimiter=",",
    retrieval_context_col_name="retrieval_context",
    retrieval_context_col_delimiter=";"
)

# timestamp
date = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

# Define your metric functions


def g_eval():

    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine whether the actual output is factually correct based on the expected output.",
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradict any facts in 'expected output'",
            "You should also heavily penalize omission of detail",
            "Vague language, or contradicting OPINIONS, are OK"
        ],
        evaluation_params=[LLMTestCaseParams.INPUT,
                           LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.8,
        model=azure_openai,
        verbose_mode=True
    )

    correctness_test_case = dataset.evaluate([correctness_metric])
    yield correctness_test_case


def summarization():

    summarization_metric = SummarizationMetric(
        threshold=0.5,
        model=azure_openai,
        verbose_mode=True,
        assessment_questions=[
            "Is the coverage score based on a percentage of 'yes' answers?",
            "Does the score ensure the summary's accuracy with the source?",
            "Does a higher score mean a more comprehensive summary?"
        ]
    )

    summarization_test_case = dataset.evaluate([summarization_metric])
    yield summarization_test_case


def faithfulness():

    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7,
        model=azure_openai,
        include_reason=True,
        verbose_mode=True
    )

    faithfulness_test_case = dataset.evaluate([faithfulness_metric])
    yield faithfulness_test_case


def answer_relevancy():

    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.8,
        model=azure_openai,
        include_reason=True,
        verbose_mode=True
    )
    answer_relevancy_test_case = dataset.evaluate([answer_relevancy_metric])
    yield answer_relevancy_test_case


def contextual_relevancy():

    contextual_relevancy_metric = ContextualRelevancyMetric(
        threshold=0.8,
        model=azure_openai,
        include_reason=True,
        verbose_mode=True
    )

    contextual_relevancy_test_case = dataset.evaluate(
        [contextual_relevancy_metric])

    yield contextual_relevancy_test_case


def contextual_precision():

    contextual_precision_metric = ContextualPrecisionMetric(
        threshold=0.8,
        model=azure_openai,
        include_reason=True,
        verbose_mode=True
    )
    contextual_precision_test_case = dataset.evaluate(
        [contextual_precision_metric])

    yield contextual_precision_test_case


def contextual_recall():

    contextual_recall_metric = ContextualRecallMetric(
        threshold=0.8,
        model=azure_openai,
        include_reason=True,
        verbose_mode=True
    )

    contextual_recall_test_case = dataset.evaluate([contextual_recall_metric])

    yield contextual_recall_test_case


def hallucination():

    hallucination_metric = HallucinationMetric(
        threshold=0.5,
        model=azure_openai,
        include_reason=True,
        verbose_mode=True
    )

    hallucination_test_case = dataset.evaluate([hallucination_metric])
    yield hallucination_test_case


def toxicity():

    toxicity_metric = ToxicityMetric(
        threshold=0.5,
        model=azure_openai,
        include_reason=True,
        verbose_mode=True
    )
    toxicity_test_case = dataset.evaluate([toxicity_metric])
    yield toxicity_test_case


def bias():

    bias_metric = BiasMetric(
        threshold=0.5,
        model=azure_openai,
        include_reason=True,
        verbose_mode=True
    )
    bias_test_case = dataset.evaluate([bias_metric])
    yield bias_test_case


def ux():

    yield "ux result", ""


# Step 1: Load the JSON file
with open('qa_Automation_config_file.json', 'r') as f:
    json_config = json.load(f)

# Step 2: Extract the EVALUATORS configuration
evaluators_config = json_config.get("EVALUATORS", {})

# Step 3: Define the metric_selector function


def metric_selector(config):
    metric_functions = {
        "g-eval": g_eval,
        "summarization": summarization,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "contextual_relevancy": contextual_relevancy,
        "contextual_precision": contextual_precision,
        "contextual_recall": contextual_recall,
        "hallucination": hallucination,
        "toxicity": toxicity,
        "bias": bias,
        "ux": ux
    }

    results = []

    for metric_name, is_enabled in config.items():
        if is_enabled:
            metric_function = metric_functions[metric_name]
            for metric_result in metric_function():
                flattened_result = {
                    "timestamp": date,
                    "metric": metric_name
                }

                # Flatten the metric_result dictionary
                if isinstance(metric_result, dict):
                    for key, value in metric_result.items():
                        flattened_result[key] = value
                elif isinstance(metric_result, list):
                    for idx, item in enumerate(metric_result):
                        if isinstance(item, dict):
                            for key, value in item.items():
                                flattened_result[f"{key}_{idx}"] = value
                        else:
                            flattened_result[f"item_{idx}"] = item

                results.append(flattened_result)

    return results


if __name__ == "__main__":
    # Example usage: Load the CSV data and use it in the metric_selector
    # data = df.to_dict(orient='records')
    results = metric_selector(evaluators_config)

    # Convert the results to a DataFrame for visualization
    results_df = pd.DataFrame(results)

    # Display the DataFrame
    print(results_df.describe())

    # Save the DataFrame to a CSV file
    results_df.to_csv("results.csv", index=False)

    print("Results saved to 'results.csv'.")


######## DATAFRAME REFACTORING ############

# Define a function to parse the TestResult string

def parse_metric_metadata(metadata_str):
    # Split the metadata string into individual metric details
    metadata_items = metadata_str.split('),')
    parsed_metadata = []

    for item in metadata_items:
        # Extract key-value pairs from each metric item
        metric_details = re.findall(r"(\w+)=('[^']*'|[^,]*)", item)
        metric_dict = {k: v.strip("'") for k, v in metric_details}
        parsed_metadata.append(metric_dict)

    return parsed_metadata


def parse_test_result_extended(test_result_str):
    # Extract success value
    success_match = re.search(r'success=(True|False)', test_result_str)
    success = success_match.group(1) == 'True' if success_match else None

    # Extract metrics_metadata
    metrics_metadata_match = re.search(
        r'metrics_metadata=\[(.*)\]', test_result_str)
    metrics_metadata = metrics_metadata_match.group(
        1) if metrics_metadata_match else None

    # Parse metrics_metadata details
    parsed_metrics_metadata = parse_metric_metadata(
        metrics_metadata) if metrics_metadata else []

    # Extract other components (input, actual_output, expected_output, etc.)
    input_match = re.search(r"input='(.*?)'", test_result_str)
    input_text = input_match.group(1) if input_match else None

    actual_output_match = re.search(r"actual_output='(.*?)'", test_result_str)
    actual_output = actual_output_match.group(
        1) if actual_output_match else None

    expected_output_match = re.search(
        r"expected_output='(.*?)'", test_result_str)
    expected_output = expected_output_match.group(
        1) if expected_output_match else None

    return {
        'success': success,
        'metrics_metadata': parsed_metrics_metadata,
        'input': input_text,
        'actual_output': actual_output,
        'expected_output': expected_output
    }

# Parse all test result items with extended parsing


def parse_all_test_results_extended(df):
    parsed_data = []
    for idx, row in df.iterrows():
        for col in df.columns:
            if col.startswith('item_'):
                parsed_result = parse_test_result_extended(row[col])
                parsed_data.append({
                    'timestamp': row['timestamp'],
                    'metric': row['metric'],
                    'item': col,
                    **parsed_result
                })
    return pd.DataFrame(parsed_data)

# Flatten the metrics_metadata into individual columns


def flatten_metrics_metadata(df):
    flattened_data = []
    for idx, row in df.iterrows():
        base_data = {
            'timestamp': row['timestamp'],
            'metric': row['metric'],
            'item': row['item'],
            'success': row['success'],
            'input': row['input'],
            'actual_output': row['actual_output'],
            'expected_output': row['expected_output']
        }
        for metric in row['metrics_metadata']:
            flattened_row = {**base_data, **metric}
            flattened_data.append(flattened_row)
    return pd.DataFrame(flattened_data)


# Load the CSV file
results_file_path = 'results.csv'
results_data = pd.read_csv(results_file_path)

# Parse the dataframe with extended parsing
parsed_df_extended = parse_all_test_results_extended(results_data)

# Flatten the extended parsed dataframe
flattened_df = flatten_metrics_metadata(parsed_df_extended)

# Display the final flattened dataframe
flattened_df.head()

# Export the flattened dataframe to a new CSV file
output_file_path = 'flattened_results.csv'
flattened_df.to_csv(output_file_path, index=False)

print("Results saved to 'flattened_results.csv'.")
