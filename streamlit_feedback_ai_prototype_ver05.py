# app.py — Google GenAI (Vertex 모드) + 튜닝/엔드포인트 지원

import streamlit as st
import pandas as pd
from datetime import datetime

from google import genai
from google.genai import types
from google.oauth2 import service_account

st.set_page_config(page_title="학습 피드백 AI", page_icon="🐸", layout="centered")

# ── Secrets ──
PROJECT_ID     = st.secrets.get("project_id", "feedback-ai-prototype-ver05")
LOCATION       = st.secrets.get("location", "us-central1")
PROJECT_NUMBER = st.secrets.get("project_number", "800102005669")  # 있으면 사용
RAW_TUNED      = (st.secrets.get("tuned_model_name") or "").strip()
ENDPOINT_NAME  = (st.secrets.get("endpoint_name") or "").strip()   # ★ 새로 추가

# 짧은 tunedModels 경로면 풀 경로로 보정
if RAW_TUNED and RAW_TUNED.startswith("tunedModels/"):
    RAW_TUNED = f"projects/{PROJECT_NUMBER}/locations/{LOCATION}/{RAW_TUNED}"

# 최종 호출에 쓸 모델 리소스(엔드포인트 우선)
MODEL_RESOURCE = ENDPOINT_NAME or RAW_TUNED
if not MODEL_RESOURCE:
    st.error("Secrets에 endpoint_name 또는 tuned_model_name 중 하나는 반드시 있어야 합니다.")
    st.stop()

# ── 인증(스코프 필수) ──
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
try:
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPES,
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

# ── 상태 ──
if "log" not in st.session_state: st.session_state.log = []
if "last_ai" not in st.session_state: st.session_state.last_ai = ""
if "last_prompt" not in st.session_state: st.session_state.last_prompt = ""
if "used_model" not in st.session_state: st.session_state.used_model = MODEL_RESOURCE
if "tuned_error" not in st.session_state: st.session_state.tuned_error = None

# ── 사이드바 ──
with st.sidebar:
    st.markdown("**환경 정보**")
    st.write(f"Project: `{PROJECT_ID}`")
    st.write(f"Location: `{LOCATION}`")
    st.write(f"endpoint_name: `{ENDPOINT_NAME or '(미설정)'}`")
    st.write(f"tuned_model_name: `{RAW_TUNED or '(미설정)'}`")
    st.write(f"effective model: `{MODEL_RESOURCE}`")
    st.write(f"scopes: `{', '.join(SCOPES)}`")
    if st.session_state.tuned_error:
        st.warning("지정 모델 호출 실패 → 베이스 모델 폴백 사용 중")
        st.exception(st.session_state.tuned_error)

# ── UI ──
st.title("🐸 독T의 학습 피드백 AI")
st.markdown("---")
user_prompt = st.text_area("학생의 상황을 자세히 입력해주세요:", height=180)

col1, col2 = st.columns(2)
with col1: gen_clicked = st.button("피드백 생성하기", use_container_width=True)
with col2: clear_clicked = st.button("화면 초기화", use_container_width=True)

if clear_clicked:
    st.session_state.last_ai = ""
    st.session_state.last_prompt = ""
    st.session_state.used_model = MODEL_RESOURCE
    st.session_state.tuned_error = None
    st.rerun()

# ── 호출 함수 ──
def call_model(model_name: str, prompt_text: str) -> str:
    resp = client.models.generate_content(
        model=model_name,  # ★ 엔드포인트 또는 튜닝 리소스
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])],
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=1024,
        ),
    )
    return resp.text or ""

# ── 실행 ──
if gen_clicked:
    if not user_prompt.strip():
        st.warning("학생의 상황을 입력해주세요.")
    else:
        with st.spinner("AI가 강사님의 철학으로 답변을 생성 중입니다..."):
            try:
                ai_text = call_model(MODEL_RESOURCE, user_prompt)
                st.session_state.used_model = MODEL_RESOURCE
                st.session_state.tuned_error = None
            except Exception as tuned_err:
                st.session_state.tuned_error = tuned_err
                # 베이스 모델(퍼블리셔 경로) 폴백
                base_model = f"projects/{PROJECT_NUMBER}/locations/{LOCATION}/publishers/google/models/gemini-2.5-pro"
                try:
                    ai_text = call_model(base_model, user_prompt)
                    st.session_state.used_model = base_model
                except Exception as base_err:
                    st.error("답변 생성 중 오류가 발생했습니다.")
                    st.exception(tuned_err)
                    st.exception(base_err)
                    raise
            st.session_state.last_ai = ai_text
            st.session_state.last_prompt = user_prompt

# ── 결과/저장 ──
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
        st.session_state.log.append({
            "timestamp": ts,
            "prompt": st.session_state.last_prompt,
            "ai_response": st.session_state.last_ai,
            "approved_response": approved.strip(),
            "used_model": st.session_state.used_model,
        })
        st.success("기록되었습니다. 아래에서 CSV로 내려받을 수 있습니다.")

# ── 로그 다운로드 ──
if st.session_state.log:
    st.markdown("---")
    st.subheader("📝 피드백 기록 다운로드")
    df = pd.DataFrame(st.session_state.log)
    csv = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("CSV 파일로 모든 기록 다운로드", data=csv, file_name="feedback_log.csv", mime="text/csv")
    if st.button("세션 로그 비우기"):
        st.session_state.log = []
        st.success("세션 로그를 비웠습니다.")
