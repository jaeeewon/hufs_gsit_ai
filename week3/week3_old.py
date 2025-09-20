# the code below is not written by me

# !pip install gradio openai-clip --quiet
# !pip install git+https://github.com/openai/CLIP.git --quiet

import os, torch, clip, csv
import gradio as gr
from PIL import Image

# CLIP 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 기준 이미지
ORIGINAL_PATH = "original.jpg"
original_img = Image.open(ORIGINAL_PATH)
original = preprocess(original_img).unsqueeze(0).to(device)
with torch.no_grad():
    base_feat = model.encode_image(original).cpu()

# 업로드 경로 + 점수 저장
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
scores = {}  # {name: best similarity score}
CSV_PATH = "ranking.csv"

#  CSV 초기화 (없으면 새로 생성)
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Name", "Score"])

# 기존 업로드 이미지 로드 & 점수 계산
for f in os.listdir(UPLOAD_DIR):
    if f.endswith((".png", ".jpg", ".jpeg")):
        name = os.path.splitext(f)[0]
        img = Image.open(os.path.join(UPLOAD_DIR, f))
        im = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(im).cpu()
        sim = torch.cosine_similarity(base_feat, feat).item()
        scores[name] = sim

def compute_ranking(text_only=True):
    """현재 점수 dict에서 순위표와 1등 이미지 반환"""
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # CSV 저장
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Name", "Score"])
        for i, (n, s) in enumerate(ranking, 1):
            writer.writerow([i, n, f"{s:.4f}"])

    # 순위표 텍스트
    rank_table = "</br>".join([f"<a target='_blank' href='/gradio_api/file=uploads/{n}.png'>{i+1}. {n} ({s:.4f})</a>" for i,(n,s) in enumerate(ranking)])

    if text_only:
        return rank_table

    # 1등 이미지
    if ranking:
        top_name = ranking[0][0]
        top_img_path = os.path.join(UPLOAD_DIR, f"{top_name}.png")
        top_img = Image.open(top_img_path)
    else:
        top_img, rank_table = None, "아직 업로드된 이미지가 없습니다."

    return top_img, rank_table

def add_and_rank(img, name):
    if not name:
        return original_img, None, "!이름(ID)을 입력하세요!"

    # 새 이미지 저장
    path = os.path.join(UPLOAD_DIR, f"{name}.png")
    img.save(path)

    # 새 점수 계산
    im = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(im).cpu()
    new_sim = torch.cosine_similarity(base_feat, feat).item()

    # 점수 비교
    msg = ""
    if name not in scores or new_sim > scores[name]:
        scores[name] = new_sim
        msg = f" {name}: 새 점수 {new_sim:.4f} (갱신됨)"
    else:
        msg = f" {name}: 새 점수 {new_sim:.4f}, 하지만 기존 점수 {scores[name]:.4f}가 더 좋아서 유지됩니다."

    # 순위표 다시 계산
    top_img, rank_table = compute_ranking(text_only=False)
    return original_img, top_img, msg + "\n\n" + rank_table

def get_top_img():
    top_img, _ = compute_ranking(text_only=False)
    return top_img

#  최초 화면 초기값 (기존 데이터 기반)
# init_top_img, init_ranking = compute_ranking()

# Gradio 인터페이스
with gr.Blocks(title="TAIM Labs Image Similarity Ranking") as demo:
    gr.Markdown("## 🖼️ Original vs 🏆 Current #1 — and Ranking")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                original_display = gr.Image(value=original_img, type="pil", label="Original", height=300, width=300)
                top_display = gr.Image(value=get_top_img, type="pil", label="🏆 Current #1", height=300, width=300)
            with gr.Row():
                upload_input = gr.Image(type="pil", label="Upload your image")
                name_input = gr.Textbox(label="Name/ID")
                submit_btn = gr.Button("Submit")
        with gr.Column():
            gr.Markdown("## Ranking Table")
            ranking_output = gr.HTML(value=compute_ranking, every=1)

    submit_btn.click(fn=add_and_rank,
                     inputs=[upload_input, name_input],
                     outputs=[original_display, top_display, ranking_output])

demo.launch(server_port=2919, server_name="0.0.0.0", allowed_paths=["uploads"])
