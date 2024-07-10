from deepeval.dataset import EvaluationDataset
from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric, ContextualRelevancyMetric, ContextualRecallMetric

dataset = EvaluationDataset()
dataset.add_test_cases_from_csv_file(
    # file_path is the absolute path to you .csv file
    file_path="example.csv",
    input_col_name="query",
    actual_output_col_name="actual_output",
    expected_output_col_name="expected_output"
)

contextual_precision_metric = ContextualPrecisionMetric()
contextual_relevancy_metric = ContextualRelevancyMetric()
contextual_recall_metric = ContextualRecallMetric()

# You can also call the evaluate() function directly
evaluate(dataset, [contextual_precision_metric, contextual_relevancy_metric, contextual_recall_metric])
