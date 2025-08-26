import streamlit as st
import pandas as pd
from datetime import datetime

import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account
import google.cloud.aiplatform as aiplatform  # 버전 확인용

# --- 디버그: 실제 로드된 secrets 키와 값 확인 ---
with st.sidebar:
    st.write("Loaded secret keys:", list(st.secrets.keys()))
    st.write("tuned_model_name:", st.secrets.get("tuned_model_name", None))


# ---------------- 페이지 기본 설정 ----------------
st.set_page_config(
    page_title="학습 피드백 AI",
    page_icon="🐸",
    layout="centered"
)

# ---------------- 설정 상수 ----------------
# 필요시 Secrets에서 덮어쓸 수 있게 기본값 + get 사용
PROJECT_ID = st.secrets.get("project_id", "feedback-ai-prototype-ver05")
LOCATION = st.secrets.get("location", "us-central1")

# ❗ 반드시 Vertex 콘솔의 Tuning 화면에서 '리소스 이름'을 복사해 넣으세요.
# 예: "tunedModels/1234567890123456789"  또는
#     "projects/feedback-ai-prototype-ver05/locations/us-central1/tunedModels/1234567890123456789"
TUNED_MODEL_NAME = st.secrets.get("projects/800102005669/locations/us-central1/models/2731304531139756032")

PROJECT_ID = st.secrets.get("project_id")
LOCATION   = st.secrets.get("location")

TUNED_MODEL_NAME = (st.secrets.get("tuned_model_name") or "").strip()
if not TUNED_MODEL_NAME:
    st.error("Secrets에 'tuned_model_name'이 없습니다. Settings → Secrets에 tunedModels/... 값을 루트에 넣고 앱을 Restart 하세요.")
    st.stop()
if not (TUNED_MODEL_NAME.startswith("tunedModels/") or "/tunedModels/" in TUNED_MODEL_NAME):
    st.error(f"tuned_model_name 형식 오류: {TUNED_MODEL_NAME}\n반드시 'tunedModels/...' 또는 'projects/.../tunedModels/...' 이어야 합니다.")
    st.stop()


# ---------------- 모델 및 인증 설정 ----------------
def load_model():
    try:
        # Streamlit Secrets 의 서비스계정 JSON으로 인증
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
    except Exception as e:
        st.error("Streamlit Secrets에 gcp_service_account 설정이 없습니다. JSON 전체를 넣어주세요.\n\n" + repr(e))
        st.stop()

    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
    except Exception as e:
        st.error("Vertex AI 초기화 실패. project/location/권한을 확인하세요.\n\n" + repr(e))
        st.stop()

    # 튜닝 모델 로드
    try:
        return GenerativeModel(TUNED_MODEL_NAME), None
    except Exception as e:
        # 베이스 모델 임시 대체 (환경 점검용)
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

# ---------------- 사이드바: 디버그/상태 ----------------
with st.sidebar:
    st.caption("환경 정보")
    st.write(f"Project: `{PROJECT_ID}`")
    st.write(f"Location: `{LOCATION}`")
    st.write(f"aiplatform: `{aiplatform.__version__}`")
    if tuned_model_error:
        st.warning("튜닝 모델 로드 실패 → 임시로 베이스 모델 사용 중")
        st.exception(tuned_model_error)

# ---------------- 앱 화면 ----------------
st.title("🐸 독T의 학습 피드백 AI")
st.markdown("---")

user_prompt = st.text_area("학생의 상황을 자세히 입력해주세요:", height=180, key="prompt_input")

col1, col2 = st.columns([1,1])
with col1:
    gen_clicked = st.button("피드백 생성하기", use_container_width=True)
with col2:
    clear_clicked = st.button("화면 초기화", use_container_width=True)

if clear_clicked:
    st.session_state.last_ai = ""
    st.session_state.last_prompt = ""
    st.rerun()

if gen_clicked:
    if not user_prompt.strip():
        st.warning("학생의 상황을 입력해주세요.")
    else:
        with st.spinner("AI가 강사님의 철학으로 답변을 생성 중입니다..."):
            try:
                # 단순 문자열 프롬프트 호출이 가장 안전
                resp = model.generate_content(
                    user_prompt,
                    generation_config={"max_output_tokens": 1024, "temperature": 0.7}
                )
                ai_text = resp.text or ""
                st.session_state.last_ai = ai_text
                st.session_state.last_prompt = user_prompt
            except Exception as e:
                st.error("답변 생성 중 오류가 발생했습니다.")
                st.exception(e)

# 결과/편집 영역
if st.session_state.last_ai:
    st.subheader("🤖 AI 초안")
    st.write(st.session_state.last_ai)

    st.markdown("### ✍️ 최종 승인용: 수정/보완해서 저장")
    approved = st.text_area(
        "필요하면 아래에서 직접 고쳐서 '기록 저장'을 누르세요.",
        value=st.session_state.last_ai,
        height=240,
        key="approved_area"
    )

    save = st.button("기록 저장", type="primary")
    if save:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.log.append({
            "timestamp": timestamp,
            "prompt": st.session_state.last_prompt,
            "ai_response": st.session_state.last_ai,
            "approved_response": approved.strip()
        })
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

    # (선택) 세션 로그 비우기
    if st.button("세션 로그 비우기"):
        st.session_state.log = []
        st.success("세션 로그를 비웠습니다.")
