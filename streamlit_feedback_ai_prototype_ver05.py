# ... 상단 생략 ...

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

# ... 인증/클라이언트 부분 동일 ...

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

# 생성 버튼에서 폴백 베이스 모델 경로를 'publishers'로 교체
if gen_clicked:
    # ...
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
    # ...
