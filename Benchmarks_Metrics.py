import os
from langchain_openai import AzureChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.tasks import MMLUTask

# Load environment variables from a .env file
load_dotenv()

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
# print(azure_openai.generate("Write me a joke"))

tasks = [MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY]
benchmark = MMLU(tasks=tasks)

results = benchmark.evaluate(model=azure_openai, batch_size=1)
print("Overall Score: ", results)
