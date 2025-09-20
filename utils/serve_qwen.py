# the code below is written by Gemini Pro and suppressed no attention mask warning
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# FastAPI 앱 초기화
app = FastAPI()

# --- 모델 로딩 ---
# 애플리케이션 시작 시 한 번만 모델을 로드합니다.
# 'device_map="auto"'는 사용 가능한 경우 GPU를 자동으로 사용하도록 설정합니다.
# 이 과정은 모델 크기에 따라 몇 분 정도 소요될 수 있습니다.
print("모델과 토크나이저를 로딩합니다...")
model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype="auto", device_map="auto"
)
print("모델 로딩이 완료되었습니다.")
# --- 모델 로딩 완료 ---


# 입력 데이터 모델 정의
class Prompt(BaseModel):
    text: str


@app.post("/generate")
def generate_text(prompt: Prompt):
    """
    입력된 텍스트(prompt)를 기반으로 모델이 새로운 텍스트를 생성하여 반환합니다.
    """
    messages = [{"role": "user", "content": prompt.text}]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=512)

    # generated_ids = [
    #     output_ids[len(input_ids) :]
    #     for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    # ]
    # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

    response = tokenizer.decode(
        generated_ids[0][model_inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    )

    return {"generated_text": response}


# --- API 서버 실행부 ---
if __name__ == "__main__":
    # Uvicorn을 사용하여 FastAPI 앱을 실행합니다.
    # host="0.0.0.0"은 로컬 네트워크 내의 다른 장치에서도 접속할 수 있게 합니다.
    uvicorn.run(app, host="0.0.0.0", port=8000)
