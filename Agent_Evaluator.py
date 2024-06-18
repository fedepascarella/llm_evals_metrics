import os
import pandas as pd
import json
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
    file_path="mock_dataset.csv",
    input_col_name="input",
    actual_output_col_name="actual_output",
    expected_output_col_name="expected_output",
    context_col_name="context",
    context_col_delimiter=",",
    retrieval_context_col_name="retrieval_context",
    retrieval_context_col_delimiter=";"
)


# timestamp
date = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

# Path to your CSV file
# file_path = 'mock_dataset.csv'

# Read the CSV file into a DataFrame
# df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
# print(df.head())

# Define your metric functions


def g_eval(data):
    for row in data:
        correctness_test_case = LLMTestCase(
            input=row["input"],
            actual_output=row["actual_output"],
            expected_output=row["expected_output"]
        )

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
            threshold=0.6,
            model=azure_openai,
            verbose_mode=True
        )

        correctness_metric.measure(correctness_test_case)
        yield correctness_metric.score, correctness_metric.reason


def summarization(data):
    for row in data:
        summarization_test_case = LLMTestCase(
            input=row["input"], actual_output=row["actual_output"])
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

        summarization_metric.measure(summarization_test_case)
        yield summarization_metric.score, summarization_metric.reason


def faithfulness():
    # for row in data:
    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7,
        model=azure_openai,
        include_reason=True,
        verbose_mode=True
    )
    # Ensure retrieval_context is a list of strings
    # retrieval_context = row["retrieval_context"] if row["retrieval_context"] else None
    # faithfulness_test_case = LLMTestCase(
    #    input=row["input"],
    #    actual_output=row["actual_output"],
    #    retrieval_context=retrieval_context
    # )
    faithfulness_test_case = dataset.evaluate([faithfulness_metric])
    # faithfulness_metric.measure(faithfulness_test_case)

    # yield faithfulness_metric.score, faithfulness_metric.reason
    yield faithfulness_test_case


def answer_relevancy(data):
    for row in data:
        answer_relevancy_metric = AnswerRelevancyMetric(
            threshold=0.7,
            model=azure_openai,
            include_reason=True,
            verbose_mode=True
        )
        answer_relevancy_test_case = LLMTestCase(
            input=row["input"],
            actual_output=row["actual_output"]
        )

        answer_relevancy_metric.measure(answer_relevancy_test_case)
        yield answer_relevancy_metric.score, answer_relevancy_metric.reason


def contextual_relevancy(data):
    for row in data:
        contextual_relevancy_metric = ContextualRelevancyMetric(
            threshold=0.7,
            model=azure_openai,
            include_reason=True,
            verbose_mode=True
        )
        retrieval_context = row["retrieval_context"] if row["retrieval_context"] else None
        contextual_relevancy_test_case = LLMTestCase(
            input=row["input"],
            actual_output=row["actual_output"],
            retrieval_context=retrieval_context
        )

        contextual_relevancy_metric.measure(contextual_relevancy_test_case)
        yield contextual_relevancy_metric.score, contextual_relevancy_metric.reason


def contextual_precision(data):
    for row in data:
        contextual_precision_metric = ContextualPrecisionMetric(
            threshold=0.7,
            model=azure_openai,
            include_reason=True,
            verbose_mode=True
        )
        retrieval_context = row["retrieval_context"] if row["retrieval_context"] else None
        contextual_precision_test_case = LLMTestCase(
            input=row["input"],
            actual_output=row["actual_output"],
            expected_output=row["expected_output"],
            retrieval_context=retrieval_context
        )

        contextual_precision_metric.measure(contextual_precision_test_case)
        yield contextual_precision_metric.score, contextual_precision_metric.reason


def contextual_recall(data):
    for row in data:
        contextual_recall_metric = ContextualRecallMetric(
            threshold=0.7,
            model=azure_openai,
            include_reason=True
        )
        retrieval_context = row["retrieval_context"] if row["retrieval_context"] else None
        contextual_recall_test_case = LLMTestCase(
            input=row["input"],
            actual_output=row["actual_output"],
            expected_output=row["expected_output"],
            retrieval_context=retrieval_context
        )

        contextual_recall_metric.measure(contextual_recall_test_case)
        yield contextual_recall_metric.score, contextual_recall_metric.reason


def hallucination(data):
    for row in data:
        hallucination_test_case = LLMTestCase(
            input=row["input"],
            actual_output=row["actual_output"],
            context=row["context"] if row["context"] else None
        )
        hallucination_metric = HallucinationMetric(
            threshold=0.5,
            model=azure_openai,
            include_reason=True,
            verbose_mode=True
        )

        hallucination_metric.measure(hallucination_test_case)
        yield hallucination_metric.score, hallucination_metric.reason


def toxicity(data):
    for row in data:
        toxicity_metric = ToxicityMetric(
            threshold=0.5,
            model=azure_openai,
            include_reason=True,
            verbose_mode=True
        )
        toxicity_test_case = LLMTestCase(
            input=row["input"],
            actual_output=row["actual_output"]
        )

        toxicity_metric.measure(toxicity_test_case)
        yield toxicity_metric.score, toxicity_metric.reason


def bias(data):
    for row in data:
        bias_metric = BiasMetric(
            threshold=0.5,
            model=azure_openai,
            include_reason=True,
            verbose_mode=True
        )
        bias_test_case = LLMTestCase(
            input=row["input"],
            actual_output=row["actual_output"]
        )

        bias_metric.measure(bias_test_case)
        yield bias_metric.score, bias_metric.reason


def ux(data):
    for row in data:
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
                results.append({
                    "timestamp": date,
                    "metric": metric_name,
                    "result": metric_result
                })

    return results


if __name__ == "__main__":
    # Example usage: Load the CSV data and use it in the metric_selector
    # data = df.to_dict(orient='records')
    results = metric_selector(evaluators_config)

    # Convert the results to a DataFrame for visualization
    results_df = pd.DataFrame(results)

    # Display the DataFrame
    print(results_df.columns)

    # Save the DataFrame to a CSV file
    results_df.to_csv("results.csv", index=False)

    print("Results saved to 'results.csv'.")
