import os, shutil, tempfile, time, json, datetime, subprocess
import gradio as gr, pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")
PASSWORD = os.getenv("PASSWORD")
__DEV__ = os.getenv("__DEV__", "0") == "1"

CONFIG_PATH = "config.json"
LEADERBOARD_PATH = "leaderboard.jsonl"

SUBMISSIONS_DIR = "submissions"
DOCKER_IMAGE = "gsit-competition:latest"
WORKDIR_IN_CONTAINER = "/workspace"

with open("problems.json", "r", encoding="utf-8") as f:
    PROBLEMS = json.load(f)
    print(f"loaded {len(PROBLEMS)} problems:")
    print(", ".join(PROBLEMS.keys()))


def default_problem_name() -> str:
    return list(PROBLEMS.keys())[0]


def load_config():
    base_problem = default_problem_name()
    base_timeout = 3.0

    if not os.path.exists(CONFIG_PATH):
        cfg = {"active_problem": base_problem, "time_limit": base_timeout}
        save_config(cfg["active_problem"], cfg["time_limit"])
        return cfg["active_problem"], cfg["time_limit"]

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}

    problem = cfg.get("active_problem", base_problem)
    if problem not in PROBLEMS:
        problem = base_problem

    try:
        timeout = float(cfg.get("time_limit", base_timeout))
    except Exception:
        timeout = base_timeout

    save_config(problem, timeout)
    return problem, timeout


def save_config(problem_name: str, time_limit: float):
    cfg = {"active_problem": problem_name, "time_limit": float(time_limit)}
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def load_leaderboard_entries():
    if not os.path.exists(LEADERBOARD_PATH):
        return []
    entries = []
    with open(LEADERBOARD_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def append_leaderboard_entry(entry: dict):
    os.makedirs(os.path.dirname(LEADERBOARD_PATH) or ".", exist_ok=True)
    with open(LEADERBOARD_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def entries_for_current_problem():
    problem, _ = load_config()
    all_entries = load_leaderboard_entries()
    filtered = [e for e in all_entries if e.get("problem_name") == problem]

    def sort_key(e):
        # 맞은 여부 -> 실행 시간 짧은 순 -> 제출 시간 빠른 순
        correct_rank = 0 if e.get("correct") else 1
        # timeout이면 exec_time=None임
        exec_time = e.get("exec_time") if e.get("exec_time") is not None else float("inf")
        submit_time = e.get("submit_time", "")
        return (correct_rank, exec_time, submit_time)

    filtered.sort(key=sort_key)
    return filtered


def build_current_leaderboard_df():
    entries = entries_for_current_problem()
    rows = []
    for i, e in enumerate(entries, start=1):
        rows.append(
            {
                "랭킹": i,
                "학번": e.get("student_id"),
                "이름": e.get("name"),
                "결과": "정답" if e.get("correct") else "시간 초과" if e.get("timeout") else "오답",
                "실행시간": e.get("exec_time"),
                "제출시간": e.get("submit_time"),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "랭킹",
                "학번",
                "이름",
                "결과",
                "실행시간",
                "제출시간",
            ]
        )
    return pd.DataFrame(rows)


def save_submission_code(student_id: str, code: str) -> str:
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{student_id}_{ts}.py"
    path = os.path.join(SUBMISSIONS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return path


def save_submission_file(py_file, student_id: str) -> str:
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{student_id}_{ts}.py"
    path = os.path.join(SUBMISSIONS_DIR, filename)
    shutil.copy(py_file.name, path)
    return path


def run_in_docker(problem_name: str, src_path: str, time_limit: float):
    tmpdir = tempfile.mkdtemp(prefix="submission_")
    try:
        shutil.copy(src_path, os.path.join(tmpdir, "main.py"))

        input_text = PROBLEMS[problem_name].get("input_text", "")
        input_filename = PROBLEMS[problem_name].get("input_filename", "input.txt")
        with open(os.path.join(tmpdir, input_filename), "w", encoding="utf-8") as f:
            f.write(input_text)

        if "upload_files" in PROBLEMS[problem_name]:
            for filename in PROBLEMS[problem_name]["upload_files"]:
                if os.path.exists(filename):
                    shutil.copy(filename, os.path.join(tmpdir, filename))

        cmd = [
            "docker",
            "run",
            "--rm",
            # "--network", "none",
            "-m",
            "256m",
            "--cpus",
            "1.0",
            "-v",
            f"{tmpdir}:{WORKDIR_IN_CONTAINER}",
            "-w",
            WORKDIR_IN_CONTAINER,
            DOCKER_IMAGE,
            "python",
            "main.py",
        ]

        start = time.monotonic()
        timed_out = False

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=time_limit,
                input="",
            )
            elapsed = time.monotonic() - start
            stdout = result.stdout.rstrip()
            stderr = result.stderr.rstrip()
        except subprocess.TimeoutExpired as e:
            timed_out = True
            elapsed = time.monotonic() - start
            stdout = (e.stdout or "") if hasattr(e, "stdout") else ""
            stderr = (e.stderr or "") if hasattr(e, "stderr") else ""
            if stderr:
                stderr += "\n"
            stderr += "[ERROR] 실행 시간 초과"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return stdout, stderr, elapsed, timed_out


def grade_submission(student_id, name, mode, py_file, code_text):
    student_id = (student_id or "").strip()
    name = (name or "").strip()

    if not student_id:
        return "학번을 입력하세요", "", ""
    if not name:
        return "이름을 입력하세요", "", ""

    problem_name, time_limit = load_config()
    expected = PROBLEMS[problem_name]["expected_output"]

    if mode == "파일 업로드":
        if py_file is None or not py_file.name.endswith(".py"):
            return ".py 파일을 업로드하세요", "", ""
        src_path = save_submission_file(py_file, student_id)
        with open(py_file.name, "r", encoding="utf-8") as f:
            code_str = f.read()
    else:
        code_str = (code_text or "").rstrip()
        if not code_str:
            return "코드를 입력하세요", "", ""
        src_path = save_submission_code(student_id, code_str)

    stdout, stderr, elapsed, timed_out = run_in_docker(problem_name, src_path, time_limit)

    if timed_out:
        correct = False
        status = f"시간 초과"
    else:
        correct = stdout == expected
        status = "정답" if correct else "오답"

    now = datetime.datetime.now().isoformat(timespec="seconds")

    entry = {
        "problem_name": problem_name,
        "student_id": student_id,
        "name": name,
        "correct": correct,
        "timeout": timed_out,
        "exec_time": None if timed_out else elapsed,
        "submit_time": now,
        "code": code_str,
        "stdout": stdout,
        "stderr": stderr,
    }
    append_leaderboard_entry(entry)

    lines = [
        f"**[채점 결과]** {status}",
        f"- 문제명: {problem_name}",
        f"- 시간 제한: {time_limit:.1f}초",
    ]
    if not timed_out:
        lines.append(f"- 실행 시간: {elapsed:.3f}초")
    summary = "\n".join(lines)

    return summary, stdout, stderr


def get_public_state():
    problem_name, time_limit = load_config()
    df = build_current_leaderboard_df()

    md = f"### 현재 진행 중인 문제\n\n" f"- 문제명: **{problem_name}**\n" f"- 시간 제한: **{time_limit:.1f}초**\n"
    return md, df


def admin_set_problem(*args):
    def _inner(password, problem_name, time_limit):
        curr_problem_name, _ = load_config()
        if password != PASSWORD:
            return (
                "잘못된 비밀번호입니다",
                curr_problem_name,
            )

        if problem_name not in PROBLEMS:
            return (
                "정의되지 않은 문제입니다",
                curr_problem_name,
            )

        try:
            time_limit = float(time_limit)
            if time_limit <= 0:
                raise ValueError
        except Exception:
            return "timeout은 양의 실수여야 합니다", curr_problem_name

        save_config(problem_name, time_limit)
        admin_msg = f"현재 문제를 '{problem_name}' (timeout={time_limit:.1f}초)로 설정했습니다"
        # admin_msg += f"\n\n- 입력: ```{PROBLEMS[problem_name].get('input_text', '')}```"
        # admin_msg += f"\n- 출력: ```{PROBLEMS[problem_name].get('expected_output', '')}```"
        return (admin_msg, problem_name)

    return *_inner(*args), build_current_leaderboard_df()


def admin_initialize_environment(password):
    def _inner(password):
        if password != PASSWORD:
            curr_problem_name, _ = load_config()
            return (
                "잘못된 비밀번호입니다",
                curr_problem_name,
            )

        if os.path.exists(SUBMISSIONS_DIR):
            for filename in os.listdir(SUBMISSIONS_DIR):
                if filename == ".gitkeep":
                    continue
                file_path = os.path.join(SUBMISSIONS_DIR, filename)
                os.remove(file_path)

        if os.path.exists(LEADERBOARD_PATH):
            os.remove(LEADERBOARD_PATH)

        default_problem = default_problem_name()
        default_timeout = 3.0
        save_config(default_problem, default_timeout)

        admin_msg = "환경을 초기화했습니다"
        return admin_msg, default_problem

    return *_inner(password), build_current_leaderboard_df()


def admin_view_submission(password, _, evt: gr.SelectData):
    if password != PASSWORD:
        return "# 잘못된 비밀번호입니다", "", ""

    if evt is None:
        return "# 행을 선택하세요", "", ""

    idx = getattr(evt, "index", None)
    if idx is None:
        return "# 선택된 행의 정보가 없습니다", "", ""

    row_idx = idx[0]

    entries = entries_for_current_problem()
    if not isinstance(row_idx, int) or row_idx < 0 or row_idx >= len(entries):
        return "# 잘못된 행 인덱스입니다", "", ""

    entry = entries[row_idx]
    problem_name, _ = load_config()

    header = (
        f"# 문제명: {problem_name}\n"
        f"# 랭킹: {row_idx + 1}\n"
        f"# 학번: {entry.get('student_id')}, 이름: {entry.get('name')}\n"
        f"# 결과: {'정답' if entry.get('correct') else '시간 초과' if entry.get('timeout') else '오답'}\n"
        f"# 제출 시간: {entry.get('submit_time')}\n"
    )

    header += f"# 실행 시간: {t_ex}\n\n" if (t_ex := entry.get("exec_time")) is not None else "\n"

    code = entry.get("code", "")
    return header + code, entry.get("stdout", ""), entry.get("stderr", "")


def refresh_all():
    info_md, df_student = get_public_state()
    df_admin = df_student
    return info_md, df_student, df_admin


with gr.Blocks(title="HUFS GSIT week12") as demo:
    problem_choices = list(PROBLEMS.keys())
    initial_problem, initial_timeout = load_config()

    gr.Markdown("# 한국외국어대학교 통번역대학원 통번역과 AI II")

    with gr.Tab("컴피티션 페이지"):
        with gr.Row():
            with gr.Column(scale=1):
                major = gr.Textbox(label="학과", placeholder="Language & AI융합전공")
                student_id = gr.Textbox(label="학번", placeholder="202402050")
                name = gr.Textbox(label="이름", placeholder="최재원")

                mode = gr.Radio(
                    label="제출 방식",
                    choices=["파일 업로드", "직접 입력"],
                    value="파일 업로드",
                )

                py_file = gr.File(label="Python 파일 업로드 (.py)", file_types=[".py"])
                code_text = gr.Code(label="코드 직접 입력 (Python)", language="python")

                submit_btn = gr.Button("채점하기", variant="primary")

            with gr.Column(scale=1):
                problem_info_md = gr.Markdown(label="현재 문제")
                result_md = gr.Markdown(label="채점 결과")
                stdout_box = gr.Textbox(label="프로그램 출력", lines=10, interactive=False)
                stderr_box = gr.Textbox(label="프로그램 오류 출력", lines=10, interactive=False)
                leaderboard_df_student = gr.Dataframe(
                    label="리더보드",
                    interactive=False,
                )

    with gr.Tab("설정 페이지"):
        gr.Markdown("### 문제 설정")
        with gr.Row():
            with gr.Column():
                password = gr.Textbox(label="비밀번호", placeholder="관리자 비밀번호를 입력하세요", type="password")
                admin_problem_select = gr.Dropdown(
                    label="진행할 문제 선택",
                    choices=problem_choices,
                    value=initial_problem,
                )
                admin_timeout_slider = gr.Slider(
                    label="시간 제한 (초)",
                    minimum=0.5,
                    maximum=60.0,
                    value=initial_timeout,
                    step=0.5,
                )
                admin_set_btn = gr.Button("문제 및 timeout 설정")
                admin_msg_md = gr.Markdown()

        if __DEV__:
            gr.Markdown("### 환경 초기화")
            with gr.Row():
                admin_reset_btn = gr.Button("환경 초기화")

        gr.Markdown("### 현재 문제 리더보드 및 제출 상세")

        with gr.Row():
            with gr.Column():
                admin_leaderboard_df = gr.Dataframe(
                    label="리더보드",
                    interactive=False,
                )
            with gr.Column():
                admin_code_view = gr.Code(label="제출된 코드", language="python")
                admin_stdout_box = gr.Textbox(label="stdout", lines=8, interactive=False)
                admin_stderr_box = gr.Textbox(label="stderr", lines=8, interactive=False)

        admin_set_btn.click(
            fn=admin_set_problem,
            inputs=[password, admin_problem_select, admin_timeout_slider],
            outputs=[admin_msg_md, admin_problem_select, admin_leaderboard_df],
        )

        admin_reset_btn.click(
            fn=admin_initialize_environment,
            inputs=[password],
            outputs=[admin_msg_md, admin_problem_select, admin_leaderboard_df],
        )

        admin_leaderboard_df.select(
            fn=admin_view_submission,
            inputs=[password, admin_leaderboard_df],
            outputs=[admin_code_view, admin_stdout_box, admin_stderr_box],
        )

    submit_btn.click(
        fn=grade_submission,
        inputs=[student_id, name, mode, py_file, code_text],
        outputs=[result_md, stdout_box, stderr_box],
    )

    demo.load(
        fn=get_public_state,
        inputs=[],
        outputs=[problem_info_md, leaderboard_df_student],
    )

    timer = gr.Timer(2.0, active=True)
    timer.tick(
        fn=refresh_all,
        inputs=[],
        outputs=[problem_info_md, leaderboard_df_student, admin_leaderboard_df],
    )

if __name__ == "__main__":
    demo.launch(server_port=1121, server_name="0.0.0.0")
