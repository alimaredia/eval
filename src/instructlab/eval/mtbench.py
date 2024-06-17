# SPDX-License-Identifier: Apache-2.0

# Local
from .evaluator import Evaluator
#from .gen_api_answer import reorg_answer_file
import instructlab.eval.gen_api_answer as gen_api_answer


class MT_Bench_Evaluator(Evaluator):
    """
    Child class of an Evaluator for Multi-turn Benchmark (MT-Bench)

    Attributes
        server_url  vLLM server endpoint
    """

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url

    def gen_answers(self, answer_file, server_url) -> str:
        """ Asks questions to model, returns path to answers"""
        gen_api_answer.run(answer_file=answer_file, model_name="instructlab/granite-7b-lab", openai_api_base=server_url)
        return answer_file

    def judge_answers(self) -> dict:
        overall_score: float = 0.0
        qa_pairs: list[tuple] = []
        payload = {"overall_score": overall_score, "qa_pairs": qa_pairs}
        return payload


class PR_Bench_Evaluator(Evaluator):
    """
    Child class of an Evaluator for PR-Bench Benchmark (PR-Bench)

    Attributes
        server_url  vLLM server endpoint
        questions   questions to be asked
    """

    def __init__(self, model_path, server_url: str, questions: str) -> None:
        super().__init__(model_path)
        self.server_url = server_url
        self.questions = questions

    def run(self) -> dict:
        overall_score = 0.0
        qa_pairs: list[tuple] = []
        payload = {"overall_score": overall_score, "qa_pairs": qa_pairs}
        return payload
