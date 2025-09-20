import gradio as gr
import json
import threading

from utils.call_qwen import LLMCall
from utils.mt_eval import Evaluator

prompt = """
[USER_PROMPT]

response must be json-like-string format. else, you will be punished.
ex: {\"translated\": [\"translated1\", \"translated2\"]}

<content>[CONTENT]</content>
"""


def complete_prompt(user_prompt: str, content: list[str]) -> str:
    str_content = "\n".join(
        [f"<content{i}>{c}</content{i}>" for i, c in enumerate(content)]
    )
    return prompt.replace("[USER_PROMPT]", user_prompt).replace(
        "[CONTENT]", str_content
    )


class Week4Prompting:
    def __init__(
        self,
        src: list[tuple[str, str]],
        max_leaderboard=10,
    ):
        self.llm = LLMCall()
        self.lock = threading.Lock()
        self.leaderboard: list[dict[str, int, float, str]] = (
            []
        )  # {name, student_id, score, prompt}
        self.max_leaderboard = max_leaderboard
        self.src = src
        self.evaluator = Evaluator()

    def _submit_prompt(self, name, student_id, prompt):
        if not name or not student_id or not prompt:
            return "three fields are required", self._get_leaderboards()

        target = [t for t, _ in self.src]
        references = [r for _, r in self.src]

        resp = self.llm.call(
            complete_prompt(
                prompt,
                target,
            )
        )
        if not (resp.startswith("{") and resp.endswith("}")):
            return "invalid llm output: " + resp, self._get_leaderboards()

        resp = json.loads(resp).get("translated", [])

        if not isinstance(resp, list) or len(resp) != len(self.src):
            return "invalid llm output: " + resp, self._get_leaderboards()

        rated = sum(self.evaluator.evaluate(references, resp)) / len(resp) * 5  # 0~5점

        with self.lock:
            self.leaderboard.append(
                {
                    "name": name,
                    "student_id": student_id,
                    "score": rated,
                    "prompt": prompt,
                }
            )
            self.leaderboard.sort(key=lambda x: x["score"], reverse=True)
            rank = (
                self.leaderboard.index(
                    next(
                        filter(
                            lambda x: x["name"] == name
                            and x["student_id"] == student_id,
                            self.leaderboard,
                        )
                    )
                )
                + 1
            )
            self.leaderboard = self.leaderboard[: self.max_leaderboard]

        str_src = "\n- ".join(target)
        str_resp = "\n- ".join(resp)
        return (
            f"score: {rated:.2f}\nrank: {rank}\nsrc: {str_src}\nres: {str_resp}",
            self._get_leaderboards(),
        )

    def _get_leaderboards(self):
        with self.lock:
            return [
                [
                    idx + 1,
                    entry["name"],
                    entry["student_id"],
                    entry["score"],
                ]
                for idx, entry in enumerate(self.leaderboard)
            ]

    def _select_leaderboard(self, evt: gr.SelectData):
        row_index = evt.index[0]

        with self.lock:
            if row_index < len(self.leaderboard):
                data = self.leaderboard[row_index]
                return f"학생: {data['name']} / {data['student_id']}\n점수: {data['score']:.2f}\n프롬프트: {data['prompt']}"
            else:
                return "unexpected case warning"

    def launch(self, **kwargs):
        with gr.Blocks(title="HUFS GSIT week4") as demo:
            gr.Markdown("# prompt well to win the prize!")

            leaderboard = gr.DataFrame(
                headers=["순위", "이름", "학번", "점수"],
                datatype=["number", "str", "number", "number"],
                value=self._get_leaderboards,
                row_count=self.max_leaderboard,
                col_count=(4, "fixed"),
                interactive=False,
                every=1,
            )

            with gr.Row():
                with gr.Column(scale=3):
                    name_input = gr.Textbox(label="이름", placeholder="이름을 입력")
                    num_input = gr.Number(label="학번", placeholder="학번을 입력")
                with gr.Column(scale=3):
                    prompt_input = gr.TextArea(
                        label="프롬포트", placeholder="프롬포트를 입력"
                    )
                    submit_button = gr.Button("프롬포트 제출", variant="primary")

            status_output = gr.TextArea(label="출력", interactive=False)

            leaderboard.select(
                fn=self._select_leaderboard, inputs=None, outputs=status_output
            )

            submit_button.click(
                fn=self._submit_prompt,
                inputs=[name_input, num_input, prompt_input],
                outputs=[status_output, leaderboard],
            )

        demo.launch(**kwargs)


if __name__ == "__main__":
    prompt_src: list[tuple[str, str]] = [
        (
            "who is the girl waiting for you in front of hufs main building?",
            "한국외국어대학교 본관 앞에서 너를 기다리는 소녀는 누구니?",
        ),
        (
            "what is the name of the library of hufs?",
            "한국외국어대학교 도서관의 이름은 무엇이니?",
        ),
        (
            "what is the name of the building where gsit is located?",
            "한국외국어대학교 gsit가 위치한 건물의 이름은 무엇이니?",
        ),
        (
            "what is the name of the building where hufs student cafeteria is located?",
            "한국외국어대학교 학생 식당이 위치한 건물의 이름은 무엇이니?",
        ),
        (
            "what is the name of the building where hufs dormitory is located?",
            "한국외국어대학교 기숙사가 위치한 건물의 이름은 무엇이니?",
        ),
    ]
    w4 = Week4Prompting(src=prompt_src)
    w4.launch(server_port=1919, server_name="0.0.0.0")
