# app.py — 학습 피드백 AI (Vertex 튜닝모델 호출 진단 모드)
import streamlit as st
import pandas as pd
from datetime import datetime

import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    Content,
    Part,
    GenerationConfig,
)
from google.oauth2 import service_account
import google.cloud.aiplatform as aiplatform

# 0) 페이지 설정
st.set_page_config(page_title="학습 피드백 AI", page_icon="🐸", layout="centered")

# 1) Secrets 로드
PROJECT_ID = st.secrets.get("project_id", "feedback-ai-prototype-ver05")
LOCATION = st.secrets.get("location", "us-central1")
PROJECT_NUMBER = st.secrets.get("project_number", None)  # 선택

_raw_model = (st.secrets.get("tuned_model_name") or "").strip()
if not _raw_model:
    st.error("Secrets에 tuned_model_name이 없습니다. tunedModels/... 또는 projects/.../tunedModels/... 값을 넣어주세요.")
    st.stop()

# 짧은 경로면 풀 경로로
if _raw_model.startswith("tunedModels/"):
    base_project = PROJECT_ID or PROJECT_NUMBER
    TUNED_MODEL_NAME = f"projects/{base_project}/locations/{LOCATION}/{_raw_model}"
else:
    TUNED_MODEL_NAME = _raw_model

if "/tunedModels/" not in TUNED_MODEL_NAME:
    st.error(f"tuned_model_name 형식 오류: {TUNED_MODEL_NAME}")
    st.stop()

# 2) 인증/초기화 + 모델 로드
def load_model():
    try:
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
    except Exception as e:
        st.error("Secrets의 [gcp_service_account]가 올바르지 않습니다.\n" + repr(e))
        st.stop()

    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=creds)
    except Exception as e:
        st.error("Vertex AI 초기화 실패.\n" + repr(e))
        st.stop()

    try:
        m = GenerativeModel(TUNED_MODEL_NAME)
        return m, None
    except Exception as e:
        # 폴백
        return GenerativeModel("gemini-1.5-flash-001"), e

model, tuned_model_error = load_model()

# 3) 사이드바 정보/디버그
with st.sidebar:
    st.markdown("**환경 정보**")
    st.write(f"Project: `{PROJECT_ID}`")
    st.write(f"Location: `{LOCATION}`")
    st.write(f"aiplatform: `{aiplatform.__version__}`")
    st.write(f"tuned_model_name: `{TUNED_MODEL_NAME}`")
    if tuned_model_error:
        st.warning("튜닝 모델 로드 실패 → 임시로 베이스 모델 사용 중")
        st.exception(tuned_model_error)

# 4) UI
st.title("🐸 독T의 학습 피드백 AI")
st.markdown("---")
user_prompt = st.text_area("학생의 상황을 자세히 입력해주세요:", height=180, key="prompt_input")

col1, col2 = st.columns([1, 1])
with col1:
    gen_clicked = st.button("피드백 생성하기", use_container_width=True)
with col2:
    clear_clicked = st.button("화면 초기화", use_container_width=True)

if clear_clicked:
    st.session_state.last_ai = ""
    st.session_state.last_prompt = ""
    st.rerun()

# 5) 호출 함수 — 3단계 전략 (어느 포맷에서 성공하는지 자동탐색)
def generate_ai_response(prompt_text: str) -> str:
    errors = []

    # (A) 가장 단순: 문자열 한 줄
    try:
        r = model.generate_content(prompt_text)
        return r.text or ""
    except Exception as e:
        errors.append(("A:string", e))

    # (B) dict 기반 contents (튜닝 샘플과 동일 구조)
    try:
        r = model.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt_text}]}]
        )
        return r.text or ""
    except Exception as e:
        errors.append(("B:dict-contents", e))

    # (C) Content/Part + GenerationConfig 객체 (권장)
    try:
        cfg = GenerationConfig(max_output_tokens=256, temperature=0.7)
        r = model.generate_content(
            [Content(role="user", parts=[Part.from_text(prompt_text)])],
            generation_config=cfg,
        )
        return r.text or ""
    except Exception as e:
        errors.append(("C:class+config", e))

    # 세 경우 모두 실패 → 상세 에러 표시
    st.error("모든 호출 포맷에서 오류가 발생했습니다. 아래 Trace를 확인하세요.")
    for tag, err in errors:
        st.markdown(f"**{tag} 실패:**")
        st.exception(err)
    raise RuntimeError("All invocation patterns failed")

# 6) 생성 버튼 처리
if gen_clicked:
    if not user_prompt.strip():
        st.warning("학생의 상황을 입력해주세요.")
    else:
        with st.spinner("AI가 강사님의 철학으로 답변을 생성 중입니다..."):
            try:
                ai_text = generate_ai_response(user_prompt)
                st.session_state.last_ai = ai_text
                st.session_state.last_prompt = user_prompt
            except Exception:
                pass

# 7) 결과/편집/저장
if st.session_state.get("last_ai"):
    st.subheader("🤖 AI 초안")
    st.write(st.session_state.last_ai)

    st.markdown("### ✍️ 최종 승인용: 수정/보완해서 저장")
    approved = st.text_area(
        "필요하면 아래에서 직접 고쳐서 '기록 저장'을 누르세요.",
        value=st.session_state.last_ai,
        height=240,
        key="approved_area",
    )

    if st.button("기록 저장", type="primary"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.log = st.session_state.get("log", [])
        st.session_state.log.append(
            {
                "timestamp": ts,
                "prompt": st.session_state.last_prompt,
                "ai_response": st.session_state.last_ai,
                "approved_response": approved.strip(),
            }
        )
        st.success("기록되었습니다. 아래에서 CSV로 내려받을 수 있습니다.")

# 8) 로그 다운로드
if st.session_state.get("log"):
    st.markdown("---")
    st.subheader("📝 피드백 기록 다운로드")
    st.caption("AI 초안과 강사님이 승인/수정한 최종 답변이 함께 저장됩니다.")
    df = pd.DataFrame(st.session_state.log)
    csv = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="CSV 파일로 모든 기록 다운로드",
        data=csv,
        file_name="feedback_log.csv",
        mime="text/csv",
    )
    if st.button("세션 로그 비우기"):
        st.session_state.log = []
        st.success("세션 로그를 비웠습니다.")
