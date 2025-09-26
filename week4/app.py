import gradio as gr
import json
import os
import threading

from utils.call_openai import LLMCall

# from utils.mt_eval import Evaluator

prompt = """
당신은 전문 번역 평가자입니다.  
다음 번역문을 평가하세요.  
평가 기준은 아래 다섯 가지입니다:

1. 정확성 (원문의 의미가 잘 전달되었는가?)  
2. 유창성 (문법적으로 자연스럽고 매끄러운가?)  
3. 용어 일관성 (전문 용어나 반복 표현이 일관되게 번역되었는가?)  
4. 문화적 적합성 (문화적 맥락이나 관용 표현이 적절히 번역되었는가?)  
5. 스타일 및 톤 (원문의 문체와 목적에 맞는가?)  

각 기준에 대해 실수 공간에서 0~5점으로 평가하고, 짧은 설명을 덧붙이세요.  
마지막에 전체 점수를 실수 공간에서 0~100점으로 산출하세요.  
각 점수를 실수 공간으로 확장해 소수점 둘째자리까지 산출하여 비판적인 입장에서 더 세밀한 평가를 할 수 있도록 하세요.

단, 원문의 언어인 한국어와 번역문이 같은 경우 최하점을 주고, 그 이유를 설명하세요.

출력은 반드시 JSONL 형식으로 하세요.  
형식 예시는 다음과 같습니다:

{"accuracy": {"score": 4.1, "comment": "의미 전달은 대체로 정확함"},  
 "fluency": {"score": 5.2, "comment": "매우 자연스러운 문장 구조"},  
 "terminology": {"score": 4.3, "comment": "대체로 일관되지만 일부 변동 있음"},  
 "cultural": {"score": 3.4, "comment": "문화적 맥락에서 조금 어색함"},  
 "style": {"score": 5.5, "comment": "원문의 문체를 잘 유지함"},  
 "overall": 85.6}  

이제 평가할 원문과 번역문을 입력합니다.  

<원문>
[CONTENT]
</원문>

<번역문>
[USER_CONTENT]
</번역문>

"""

str_content = """
창밖의 하늘은 잿빛으로 가라앉아 있었다.
나는 손끝에 닿는 바람 속에서 오래된 목소리를 들었다.
지워지지 않는 그림자가 가슴 속을 헤매고,
그 무게에 숨이 막히는 듯했다.
"""

pid = os.getpid()


def complete_prompt(user_content: str) -> str:
    return prompt.replace("[USER_CONTENT]", user_content).replace(
        "[CONTENT]", str_content
    )


def check_valid(entry: dict) -> bool:
    required_keys = {
        "accuracy",
        "fluency",
        "terminology",
        "cultural",
        "style",
        "overall",
    }
    if set(entry.keys()) != required_keys:
        return False
    for key in required_keys - {"overall"}:
        if not isinstance(entry[key], dict):
            return False
        if set(entry[key].keys()) != {"score", "comment"}:
            return False
        if not (
            isinstance(entry[key]["score"], float) and 0 <= entry[key]["score"] <= 5
        ):
            return False
        if not isinstance(entry[key]["comment"], str):
            return False
    if not (isinstance(entry["overall"], float) and 0 <= entry["overall"] <= 100):
        return False

    return True


class Week4Prompting:
    def __init__(self, max_leaderboard=10, placeholders=[]):
        self.llm = LLMCall()
        self.lock = threading.Lock()
        self.leaderboard: list[dict[str, int, float, str]] = (
            []
        )  # {name, student_id, score, prompt}
        self.max_leaderboard = max_leaderboard
        # self.evaluator = Evaluator()
        for entry in placeholders:
            name, student_id, content = entry
            self._submit_content(name, student_id, content)

    def _submit_content(self, name, student_id, content):
        if not name or not student_id or not content:
            return "rejected: three fields are required", self._get_leaderboards()

        # check duplicate content
        for entry in self.leaderboard:
            if entry["content"] == content:
                return (
                    f"rejected: duplicate content found | {entry['name']} / {entry['student_id']}",
                    self._get_leaderboards(),
                )

        resp = (
            self.llm.call(
                complete_prompt(
                    content,
                )
            )
            .replace("```jsonl", "")
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
        if not (resp.startswith("{") and resp.endswith("}")):
            return (
                "[조교에게 문의주세요] invalid llm output: " + resp,
                self._get_leaderboards(),
            )

        resp = json.loads(resp)

        """
        {
            "accuracy": {"score": 4, "comment": "의미 전달은 대체로 정확함"},
            "fluency": {"score": 5, "comment": "매우 자연스러운 문장 구조"},
            "terminology": {"score": 4, "comment": "대체로 일관되지만 일부 변동 있음"},
            "cultural": {"score": 3, "comment": "문화적 맥락에서 조금 어색함"},
            "style": {"score": 5, "comment": "원문의 문체를 잘 유지함"},
            "overall": 85,
        }
        """
        # 응답이 올바른지
        if not check_valid(resp):
            return (
                "[조교에게 문의주세요] invalid llm output: " + str(resp),
                self._get_leaderboards(),
            )

        # rated = sum(self.evaluator.evaluate(references, resp)) / len(resp) * 5  # 0~5점

        with self.lock:
            self.leaderboard.append(
                {
                    "name": name,
                    "student_id": student_id,
                    "score": resp["overall"],
                    "content": content,
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

        rtn_txt = f"평가 점수: {resp['overall']:.2f}점 ({rank}등)\n"
        rtn_txt += "세부 평가:\n"
        for key in ["accuracy", "fluency", "terminology", "cultural", "style"]:
            rtn_txt += f"- {key}: {resp[key]['score']}점 ({resp[key]['comment']})\n"

        with open(f"leaderboard_{pid}.json", "w", encoding="utf-8") as f:
            json.dump(self.leaderboard, f, indent=4, ensure_ascii=False)

        return (
            rtn_txt,
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
                return f"학생: {data['name']} / {data['student_id']}\n점수: {data['score']:.2f}\n제출된 번역\n{data['content']}"
            else:
                return "unexpected case warning"

    def launch(self, **kwargs):
        with gr.Blocks(title="HUFS GSIT week4") as demo:
            gr.Markdown("# prompt well to win the prize!")

            with gr.Row():
                with gr.Column(scale=3):
                    name_input = gr.Textbox(label="이름", placeholder="이름을 입력")
                    num_input = gr.Number(label="학번", placeholder="학번을 입력")
                with gr.Column(scale=3):
                    prompt_input = gr.TextArea(
                        label="번역", placeholder="번역된 내용을 입력"
                    )
                    submit_button = gr.Button("번역 제출", variant="primary")

            gr.Textbox(
                label="원문",
                value=str_content.strip(),
                interactive=False,
            )
            status_output = gr.TextArea(label="출력", interactive=False)

            leaderboard = gr.DataFrame(
                headers=["순위", "이름", "학번", "점수"],
                datatype=["number", "str", "number", "number"],
                value=self._get_leaderboards,
                row_count=self.max_leaderboard,
                col_count=(4, "fixed"),
                interactive=False,
                every=1,
            )

            leaderboard.select(
                fn=self._select_leaderboard, inputs=None, outputs=status_output
            )

            submit_button.click(
                fn=self._submit_content,
                inputs=[name_input, num_input, prompt_input],
                outputs=[status_output, leaderboard],
            )

        demo.launch(**kwargs)


if __name__ == "__main__":
    placeholders = [
        [
            "Papago",
            2025,
            """The sky outside the window was gray.
I heard an old voice in the wind at my fingertips.
An indelible shadow wanders through your heart,
I felt suffocated by the weight.""",
        ],
        [
            "Google Translate",
            2025,
            """The sky outside the window had sunk to a gray ash.
I heard an ancient voice in the wind at my fingertips.
An indelible shadow wandered through my heart,
and its weight seemed to suffocate me.""",
        ],
    ]

    w4 = Week4Prompting(placeholders=placeholders)
    w4.launch(server_port=1919, server_name="0.0.0.0")
