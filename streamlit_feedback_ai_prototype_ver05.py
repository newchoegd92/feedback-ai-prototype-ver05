# app.py — 학습 피드백 AI (Google GenAI + Vertex 튜닝 모델)

import streamlit as st
import pandas as pd
from datetime import datetime

from google import genai
from google.genai import types
from google.oauth2 import service_account

# ───────────────── 기본 설정 ─────────────────
st.set_page_config(page_title="학습 피드백 AI", page_icon="🐸", layout="centered")

# ───────────────── Secrets ─────────────────
PROJECT_ID    = st.secrets.get("project_id", "feedback-ai-prototype-ver05")
LOCATION      = st.secrets.get("location", "us-central1")
PROJECT_NUMBER = st.secrets.get("project_number", None)  # 선택

RAW_TUNED = (st.secrets.get("tuned_model_name") or "").strip()
if not RAW_TUNED:
    st.error("Secrets에 tuned_model_name이 없습니다.")
    st.stop()

# 짧은 경로면 풀 경로로, 그리고 가능하면 프로젝트 '넘버'를 우선 사용
if RAW_TUNED.startswith("tunedModels/"):
    base_project = PROJECT_NUMBER or PROJECT_ID
    TUNED_MODEL_NAME = f"projects/{base_project}/locations/{LOCATION}/{RAW_TUNED}"
else:
    TUNED_MODEL_NAME = RAW_TUNED

# ───────────────── 인증 & 클라이언트 ─────────────────
# ★ OAuth scope 필수 (invalid_scope 방지)
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

try:
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPES,  # ← 중요
    )
except Exception as e:
    st.error("Secrets의 [gcp_service_account]가 올바르지 않습니다.\n" + repr(e))
    st.stop()

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    credentials=credentials,
)

# ───────────────── 세션 상태 ─────────────────
if "log" not in st.session_state:
    st.session_state.log = []
if "last_ai" not in st.session_state:
    st.session_state.last_ai = ""
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""
if "used_model" not in st.session_state:
    st.session_state.used_model = TUNED_MODEL_NAME
if "tuned_error" not in st.session_state:
    st.session_state.tuned_error = None

# ───────────────── 사이드바 ─────────────────
with st.sidebar:
    st.markdown("**환경 정보**")
    st.write(f"Project: `{PROJECT_ID}`")
    st.write(f"Location: `{LOCATION}`")
    st.write(f"tuned_model_name: `{TUNED_MODEL_NAME}`")
    st.write(f"scopes: `{', '.join(SCOPES)}`")
    if st.session_state.tuned_error:
        st.warning("튜닝 모델 호출 실패 → 베이스 모델 폴백 사용 중")
        st.exception(st.session_state.tuned_error)

# ───────────────── UI ─────────────────
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
    st.session_state.used_model = TUNED_MODEL_NAME
    st.session_state.tuned_error = None
    st.rerun()

# ───────────────── 호출 함수 ─────────────────
def call_model(model_name: str, prompt_text: str) -> str:
    resp = client.models.generate_content(
        model=model_name,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])],
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=1024,
        ),
    )
    return resp.text or ""


# ───────────────── 생성 버튼 처리 ─────────────────
if gen_clicked:
    if not user_prompt.strip():
        st.warning("학생의 상황을 입력해주세요.")
    else:
        with st.spinner("AI가 강사님의 철학으로 답변을 생성 중입니다..."):
            try:
                ai_text = call_model(TUNED_MODEL_NAME, user_prompt)
                st.session_state.used_model = TUNED_MODEL_NAME
                st.session_state.tuned_error = None
            except Exception as tuned_err:
                st.session_state.tuned_error = tuned_err
                try:
                    base_project = PROJECT_NUMBER or PROJECT_ID
                    base_model = f"projects/{base_project}/locations/{LOCATION}/publishers/google/models/gemini-2.5-pro"
                    ai_text = call_model(base_model, user_prompt)
                    st.session_state.used_model = base_model
                except Exception as base_err:
                    st.error("답변 생성 중 오류가 발생했습니다.")
                    st.exception(tuned_err)
                    st.exception(base_err)
                raise
            st.session_state.last_ai = ai_text
            st.session_state.last_prompt = user_prompt

# ───────────────── 결과/편집/저장 ─────────────────
if st.session_state.last_ai:
    st.subheader("🤖 AI 초안")
    st.caption(f"사용한 모델: `{st.session_state.used_model}`")
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
        st.session_state.log.append(
            {
                "timestamp": ts,
                "prompt": st.session_state.last_prompt,
                "ai_response": st.session_state.last_ai,
                "approved_response": approved.strip(),
                "used_model": st.session_state.used_model,
            }
        )
        st.success("기록되었습니다. 아래에서 CSV로 내려받을 수 있습니다.")

# ───────────────── 로그 다운로드 ─────────────────
if st.session_state.log:
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
