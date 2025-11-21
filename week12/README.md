# WEEK12: Vibe Coding Competition

## Overview

수강생 분들께서 파이썬 파일을 업로드하면, 이를 샌드박스 환경에서 실행하고 평가하는 시스템을 구현합니다.

## 사용 방법

- 패키지를 사용해야 할 수 있으므로, 도커 이미지를 빌드하여 사용합니다. \
`app.py` 코드는 `gsit-competition:latest` 이미지가 필요합니다.
- `problems.json` 파일에 문제를 정의합니다. \
구조는 다음을 따라야 합니다.
    ```json
    {
        "문제명": {
            "expected_output": "예상 출력값 (정답)",
            "input_text": "input.txt 파일에 들어갈 내용",
            "upload_files": [
                "실행시 필요한 추가 파일명"
            ]
        }, ...
    }
    ```

### 도커 이미지 빌드 명령어

```bash
docker build -t gsit-competition:latest .
```

### 바이브 코딩 컴피티션 실행 명령어

```bash
python app.py
```
