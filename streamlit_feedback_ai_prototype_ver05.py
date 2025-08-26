# app.py
# ================== 학습 피드백 AI (Streamlit + Vertex AI 튜닝 모델) ==================
# 필수: Settings → Secrets 에 아래 키들이 있어야 합니다.
#   project_id, location(=us-central1), tuned_model_name(풀 경로 권장)
#   [gcp_service_account]  ← 서비스계정 JSON 원문
# --------------------------------------------------------------------

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

# ---------------- 페이지 기본 설정 (Streamlit의 첫 호출이어야 함) ----------------
st.set_page_config(page_title="학습 피드백 AI", page_icon="🐸", layout="centered")

# ---------------- Secrets 로드 & 기본값 ----------------
PROJECT_ID = st.secrets.get("project_id", "feedback-ai-prototype-ver05")
LOCATION = st.secrets.get("location", "us-central1")
PROJECT_NUMBER = st.secrets.get("project_number", None)  # 선택

_raw_model_name = (st.secrets.get("tuned_model_name") or "").strip()
if not _raw_model_name:
    st.error("Secrets에 'tuned_model_name'이 없습니다. Settings → Secrets에 tunedModels/... 값을 추가하세요.")
    st.stop()

# 짧은 형태(tunedModels/123...)가 오면 풀 경로로 정규화
if _raw_model_name.startswith("tunedModels/"):
    base_project = PROJECT_ID or PROJECT_NUMBER
    TUNED_MODEL_NAME = f"projects/{base_project}/locations/{LOCATION}/{_raw_model_name}"
else:
    TUNED_MODEL_NAME = _raw_model_name

if "/tunedModels/" not in TUNED_MODEL_NAME:
    st.error(f"tuned_model_name 형식 오류: {TUNED_MODEL_NAME}\n반드시 'tunedModels/...' 또는 'projects/.../tunedModels/...' 이어야 합니다.")
    st.stop()

# ---------------- 모델 및 인증 설정 ----------------
def load_model():
    # 1) 인증
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
    except Exception as e:
        st.error("Secrets의 [gcp_service_account] 설정이 올바르지 않습니다.\n" + repr(e))
        st.stop()

    # 2) Vertex 초기화
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
    except Exception as e:
        st.error("Vertex AI 초기화 실패: project/location/권한을 확인하세요.\n" + repr(e))
        st.stop()

    # 3) 튜닝 모델 로드
    try:
        tuned = GenerativeModel(TUNED_MODEL_NAME)
        return tuned, None
    except Exception as e:
        # 폴백: 베이스 모델로라도 동작 확인 가능하게
        fallback = GenerativeModel("gemini-1.5-flash-001")
        return fallback, e

model, tuned_model_error = load_model()

# ---------------- 세션 상태 ----------------
if "log" not in st.session_state:
    st.session_state.log = []
if "last_ai" not in st.session_state:
    st.session_state.last_ai = ""
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""

# ---------------- 사이드바: 환경/디버그 ----------------
with st.sidebar:
    st.caption("환경 정보")
    st.write(f"Project: `{PROJECT_ID}`")
    st.write(f"Location: `{LOCATION}`")
    st.write(f"aiplatform: `{aiplatform.__version__}`")
    st.write(f"tuned_model_name: `{TUNED_MODEL_NAME}`")
    if tuned_model_error:
        st.warning("튜닝 모델 로드 실패 → 임시로 베이스 모델 사용 중")
        st.exception(tuned_model_error)

# ---------------- UI ----------------
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

# ---------------- 호출 함수 ----------------
def generate_ai_response(prompt_text: str) -> str:
    """
    1차: 최소 포맷 호출 (가장 호환성이 높음)
    2차: GenerationConfig 객체만 사용해 옵션 적용
    """
    # 1) 최소 포맷
    try:
        resp = model.generate_content([Content(role="user", parts=[Part.from_text(prompt_text)])])
        return resp.text or ""
    except Exception:
        pass

    # 2) 정식 객체 기반 설정 (dict 금지)
    cfg = GenerationConfig(max_output_tokens=512, temperature=0.7)
    resp = model.generate_content(
        [Content(role="user", parts=[Part.from_text(prompt_text)])],
        generation_config=cfg,
    )
    return resp.text or ""

# ---------------- 생성 버튼 처리 ----------------
if gen_clicked:
    if not user_prompt.strip():
        st.warning("학생의 상황을 입력해주세요.")
    else:
        with st.spinner("AI가 강사님의 철학으로 답변을 생성 중입니다..."):
            try:
                ai_text = generate_ai_response(user_prompt)
                st.session_state.last_ai = ai_text
                st.session_state.last_prompt = user_prompt
            except Exception as e:
                st.error("답변 생성 중 오류가 발생했습니다.")
                st.exception(e)

# ---------------- 결과/편집/저장 ----------------
if st.session_state.last_ai:
    st.subheader("🤖 AI 초안")
    st.write(st.session_state.last_ai)

    st.markdown("### ✍️ 최종 승인용: 수정/보완해서 저장")
    approved = st.text_area(
        "필요하면 아래에서 직접 고쳐서 '기록 저장'을 누르세요.",
        value=st.session_state.last_ai,
        height=240,
        key="approved_area",
    )

    save = st.button("기록 저장", type="primary")
    if save:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.log.append(
            {
                "timestamp": timestamp,
                "prompt": st.session_state.last_prompt,
                "ai_response": st.session_state.last_ai,
                "approved_response": approved.strip(),
            }
        )
        st.success("기록되었습니다. 아래에서 CSV로 내려받을 수 있습니다.")

# ---------------- 로그 다운로드 ----------------
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
