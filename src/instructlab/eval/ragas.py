# # SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, TypedDict

# Third Party
from langchain_community.chat_models import ChatOpenAI
from openai import Client as OpenAIClient
from openai.types.chat import ChatCompletionMessageParam
from pandas import DataFrame, read_json
from pydantic import BaseModel, ConfigDict, Field
from ragas.evaluation import EvaluationDataset, EvaluationResult, RunConfig, evaluate
from ragas.metrics import Metric
from ragas.metrics._domain_specific_rubrics import (  # the rubrics we must instantiate are located inside of a file marked as private
    RubricsScore,
    SingleTurnPrompt,
)

# Local
from .evaluator import Evaluator
from .logger_config import setup_logger

logger = setup_logger(__name__)

OLD_DEFAULT_WITH_REFERENCE_RUBRICS = {
    "score1_description": "The response is incorrect, irrelevant, or does not align with the ground truth.",
    "score2_description": "The response partially matches the ground truth but includes significant errors, omissions, or irrelevant information.",
    "score3_description": "The response generally aligns with the ground truth but may lack detail, clarity, or have minor inaccuracies.",
    "score4_description": "The response is mostly accurate and aligns well with the ground truth, with only minor issues or missing details.",
    "score5_description": "The response is fully accurate, aligns completely with the ground truth, and is clear and detailed.",
}


class Sample(TypedDict):
    """
    TypedDict of a sample that we accept when doing eval with Ragas.
    We specifically use TypedDict here to be flexible with the input data we accept.
    """

    # question
    user_input: str

    # model answer
    response: Optional[str]

    # golden answer
    reference: str


# default system prompt we'll use when none is provided. Make it private as we don't intend this to be a public object
_DEFAULT_SYSTEM_PROMPT = """You are an advanced AI assistant designed to provide precise and accurate information.
Your primary goal is to answer queries with the most up-to-date and factual information available.
Focus on delivering clear, concise, and correct responses.
If you're uncertain about any aspect of the query, state your level of confidence and provide the most accurate information you can.
Your responses should prioritize accuracy over all other considerations."""

DEFAULT_SEED = 1337
DEFAULT_JUDGE_MODEL = "gpt-4o"


class ModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    # name of the model to use.
    model_name: str

    # The system prompt to be used when applying the chat template.
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT

    # "model randomness" aka likelihood of sampling something other than the likeliest token
    temperature: float = Field(default=0.0, le=1.0, ge=0.0)

    # Max amount of tokens to generate.
    max_tokens: int = 768

    # Random seed for reproducibility. Caution: this isn't supported by all model serving runtimes.
    seed: int = 42


class RagasEvaluator(Evaluator):
    # most basic implementation, we just assume that the user will bring the existing model responses
    name = "ragas"

    def __init__(
        self,
        student_model: ModelConfig | None = None,
        run_config: RunConfig | None = None,
        student_openai_client: OpenAIClient | None = None,
        judge_model_name: str = DEFAULT_JUDGE_MODEL,
        judge_openai_api_key: str | None = None,
    ):
        self.student_model = student_model
        self.run_config = run_config
        self.student_openai_client = student_openai_client
        self.judge_model_name = judge_model_name
        self.judge_openai_api_key = judge_openai_api_key

    @staticmethod
    def validate_dataset(df: DataFrame):
        """
        Validates whether or not the given `df` is a valid dataset of `Sample` objects.

        Args:
            df (DataFrame):            DataFrame containing the dataset to be evaluated.
        """
        required_keys = {"user_input", "reference", "response"}

        columns_list = set(df.columns)
        if not columns_list.issubset(required_keys):
            raise ValueError(
                f"Dataset can only have the following keys: {', '.join(required_keys)}. Keys provided were: {', '.join(columns_list)}"
            )

    def run(
        self,
        dataset: List[Sample] | DataFrame,
        run_config: RunConfig | None = None,
        judge_model_name: str | None = None,
        judge_openai_api_key: str | None = None,
    ) -> EvaluationResult:
        """
        Evaluates the quality of model responses against a graded rubric.

        When the `dataset` lacks the `response` field, then `student_model` must be provided
        in order to generate the answers.

        Args:
            dataset (List[Sample] | Path):
                Can be either a list of `Sample` objects or a path to a jsonl file containing
                records matching `Sample`.
                generate the responses from the given list of questions.
            run_config (RunConfig | None, optional):
                Configuration to use when running evaluations. If none is provided, then
                a default one is created containing extremely permissive settings when handling
                timeouts. This is because by default, OpenAI tier-1 usage accounts have very high
                rate limits resulting in heavy throttling during evaluations.
            judge_model_name (str | None, optional):
                Name of the OpenAI model to use as the judge model. Defaults to "gpt-4o" when none is specified.
            judge_openai_api_key (str | None, optional):
                The API key to use for evaluating the given dataset. When this isn't provided, `OPENAI_API_KEY` is read instead.


        Returns:
            EvaluationResult: The results of all evaluations performed by Ragas
        """
        judge_model_name = (
            judge_model_name if judge_model_name else self.judge_model_name
        )
        judge_openai_api_key = (
            judge_openai_api_key if judge_openai_api_key else self.judge_openai_api_key
        )
        run_config = run_config if run_config else self.run_config

        # ensure we are in the dataframe format
        input_df = dataset
        if isinstance(dataset, list):
            input_df = DataFrame(dataset)
        elif not isinstance(dataset, DataFrame):
            raise TypeError(f"invalid type of dataset: {type(dataset)}")

        # this should never happen, but pylint is not smart enough to detect it
        if TYPE_CHECKING:
            assert input_df is not None

        # ensure the dataset is in the format we expect it
        # this looks similar to validate_dataset but here we want an exact match, not a subset
        required_keys = {"user_input", "reference", "response"}
        columns = set(input_df.columns)
        if columns != required_keys:
            raise ValueError(
                f"Input Dataset can only have the following keys: {', '.join(required_keys)}. Keys provided were: {', '.join(columns)}"
            )


        if not run_config:
            # we set extreme timeout/retry values by default since OpenAI tier-1 rate limits
            # are horrible and will result in half of our evaluation results being NaN or 0
            run_config = RunConfig(
                max_retries=120,
                max_wait=7200,
                seed=DEFAULT_SEED,
                timeout=3600,
            )

        metrics = self._get_metrics()
        evaluation_ds = EvaluationDataset.from_pandas(input_df)

        # we will be using gpt-4o for the foreseeable future, we hardcode this
        # for consistency of answers

        critic_lm = ChatOpenAI(model=judge_model_name, api_key=judge_openai_api_key)
        results = evaluate(
            dataset=evaluation_ds,
            batch_size=4,
            run_config=run_config,
            llm=critic_lm,
            metrics=metrics,
            show_progress=True,
        )
        return results

    @staticmethod
    def generate_answers_from_model(
        questions: DataFrame,
        model_config: ModelConfig,
        openai_client: OpenAIClient,
    ) -> DataFrame:
        """
        Given a DataFrame containing `user_input` columns, generates responses from the given model
        and returns a new DataFrame containing its answers in the `response` column.

        Args:
            questions: (DataFrame):
                Questions and refernce answers to be returned with the responses from the model
            model_config: (ModelConfig):
                Configuration settings for the model when getting responses.
            openai_client (openai.Client | None, optional):
                The client to use when generating questions from the model, must be compatible with the OpenAI API.
        Returns:
            DataFrame with user_input, reference, and response columns. Responses for the user_input from the model
        """
        # initialize response to write into
        updated_df = questions.copy()
        updated_df["response"] = ""

        for i, qna in updated_df.iterrows():
            messages: List[ChatCompletionMessageParam] = [
                {
                    "role": "system",
                    "content": model_config.system_prompt,
                },
                {"role": "user", "content": qna["user_input"]},
            ]
            response = openai_client.chat.completions.create(
                messages=messages,
                model=model_config.model_name,
                # specify the seed so we can at least try to have some reproducibility when the clients support it
                seed=model_config.seed,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
            )
            updated_df.at[i, "response"] = response.choices[0].message.content
        return updated_df

    @staticmethod
    def _get_metrics() -> List[Metric]:
        return [
            RubricsScore(
                rubrics=OLD_DEFAULT_WITH_REFERENCE_RUBRICS,
            )
        ]