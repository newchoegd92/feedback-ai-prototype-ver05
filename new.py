# app.py  — Admin (리뷰/승인/내보내기) 앱
# -----------------------------------------------------------
# ✅ 필요한 Secrets (Streamlit → Settings → Secrets)
# project_id = "feedback-ai-prototype-ver05"
# location   = "us-central1"
# endpoint_name = "projects/800102005669/locations/us-central1/endpoints/6803710882468593664"
#
# # raw(사용자 제출 저장소)
# raw_bucket_name = "feedback-proto-ai-raw"
# raw_prefix      = "raw_submissions"
#
# # curated(관리자 승인 저장소) — 별도 버킷을 쓰면 버킷명 변경
# cur_bucket_name = "feedback-ai-curated"   # 또는 raw 버킷과 동일 이름
# cur_prefix      = "curated"
#
# [gcp_service_account]
# ... 서비스계정 JSON 원문 ...
# -----------------------------------------------------------

import io
import json
import uuid
from datetime import datetime, date
from typing import List, Dict, Any

import pandas as pd
import streamlit as st

from google.oauth2 import service_account
from google.cloud import storage
from google import genai
from google.genai import types

# -------------------- 페이지/상수 --------------------
st.set_page_config(page_title="개구리 학습 피드백 (Admin)", page_icon="🐸", layout="wide")

PROJECT_ID = st.secrets.get("project_id")
LOCATION   = st.secrets.get("location", "us-central1")
ENDPOINT   = st.secrets.get("endpoint_name", "").strip()

RAW_BUCKET = st.secrets.get("raw_bucket_name", "")
RAW_PREFIX = (st.secrets.get("raw_prefix") or "raw_submissions").strip().strip("/")

CUR_BUCKET = st.secrets.get("cur_bucket_name", RAW_BUCKET or "")
CUR_PREFIX = (st.secrets.get("cur_prefix") or "curated").strip().strip("/")

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# -------------------- 인증/클라이언트 --------------------
try:
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=SCOPES
    )
except Exception as e:
    st.error("Secrets의 [gcp_service_account] 설정을 확인하세요.")
    st.stop()

try:
    client = genai.Client(
        vertexai=True, project=PROJECT_ID, location=LOCATION, credentials=credentials
    )
    storage_client = storage.Client(project=PROJECT_ID, credentials=credentials)
except Exception as e:
    st.error("클라이언트 초기화 실패:\n" + repr(e))
    st.stop()

# -------------------- 유틸 --------------------
def gcs_upload_bytes(bucket: str, path: str, data: bytes, content_type: str):
    b = storage_client.bucket(bucket).blob(path)
    b.cache_control = "no-cache"
    b.upload_from_file(io.BytesIO(data), size=len(data), content_type=content_type)

def upload_json(bucket: str, path: str, obj: Dict[str, Any]):
    gcs_upload_bytes(bucket, path, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"), "application/json")

def read_json(bucket: str, path: str) -> Dict[str, Any]:
    b = storage_client.bucket(bucket).blob(path)
    return json.loads(b.download_as_bytes())

def delete_object(bucket: str, path: str):
    storage_client.bucket(bucket).blob(path).delete()

@st.cache_data(ttl=60)
def list_keys(bucket: str, prefix: str) -> List[str]:
    """버킷의 prefix 하위 .json 키를 최신순으로 반환 (캐시 60초)"""
    if not bucket or not prefix:
        return []
    blobs = storage_client.list_blobs(bucket, prefix=f"{prefix}/")
    keys = [b.name for b in blobs if b.name.endswith(".json")]
    keys.sort(reverse=True)
    return keys

def key_date(key: str) -> str:
    # prefix/YYYY-MM-DD/file.json → YYYY-MM-DD
    parts = key.split("/")
    if len(parts) >= 3:
        return parts[1]
    return ""

def filter_keys_by_date(keys: List[str], start: date, end: date) -> List[str]:
    s = start.strftime("%Y-%m-%d")
    e = end.strftime("%Y-%m-%d")
    out = []
    for k in keys:
        kd = key_date(k)
        if kd and (s <= kd <= e):
            out.append(k)
    return out

def load_entries(bucket: str, keys: List[str]) -> List[Dict[str, Any]]:
    out = []
    for k in keys:
        try:
            d = read_json(bucket, k)
            d["_bucket"] = bucket
            d["_key"] = k
            out.append(d)
        except Exception:
            # 읽기 실패 항목은 스킵
            pass
    return out

def curated_key_from_raw(raw_key: str) -> str:
    parts = raw_key.split("/", 2)  # [raw_prefix, yyyy-mm-dd, file]
    if len(parts) >= 3:
        return f"{CUR_PREFIX}/{parts[1]}/{parts[2]}"
    ts = datetime.utcnow().strftime("%Y-%m-%d")
    return f"{CUR_PREFIX}/{ts}/{uuid.uuid4().hex[:10]}.json"

def call_model(model_name: str, prompt_text: str) -> str:
    resp = client.models.generate_content(
        model=model_name,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])],
        config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=1024),
    )
    return resp.text or ""

def to_jsonl_lines(entries: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for e in entries:
        prompt = (e.get("prompt") or "").strip()
        # approved가 있으면 우선 사용, 없으면 ai_response 사용
        out = (e.get("approved_response") or e.get("ai_response") or "").strip()
        if not prompt or not out:
            continue
        obj = {
            "contents": [
                {"role": "user",  "parts": [{"text": prompt}]},
                {"role": "model", "parts": [{"text": out}]},
            ]
        }
        lines.append(json.dumps(obj, ensure_ascii=False))
    return lines

def to_dataframe(entries: List[Dict[str, Any]]) -> pd.DataFrame:
    cols = [
        "timestamp","prompt","ai_response","approved_response","approved_by","approved_at",
        "review_notes","used_model","source_raw_bucket","source_raw_key","_bucket","_key"
    ]
    rows = []
    for e in entries:
        rows.append({
            "timestamp": e.get("timestamp"),
            "prompt": e.get("prompt"),
            "ai_response": e.get("ai_response"),
            "approved_response": e.get("approved_response"),
            "approved_by": e.get("approved_by"),
            "approved_at": e.get("approved_at"),
            "review_notes": e.get("review_notes"),
            "used_model": e.get("used_model"),
            "source_raw_bucket": e.get("source_raw_bucket"),
            "source_raw_key": e.get("source_raw_key"),
            "_bucket": e.get("_bucket"),
            "_key": e.get("_key"),
        })
    return pd.DataFrame(rows, columns=cols)

def contains_keyword(e: Dict[str, Any], kw: str) -> bool:
    if not kw:
        return True
    kw = kw.lower()
    for field in ("prompt","ai_response","approved_response","review_notes"):
        v = (e.get(field) or "")
        if kw in v.lower():
            return True
    return False

# -------------------- 사이드바 --------------------
with st.sidebar:
    st.markdown("### 환경 정보")
    st.write(f"Project: `{PROJECT_ID}`")
    st.write(f"Location: `{LOCATION}`")
    st.write(f"Endpoint: `{ENDPOINT or '(미설정)'}`")
    st.write("---")
    st.write(f"raw: `gs://{RAW_BUCKET}/{RAW_PREFIX}`")
    st.write(f"curated: `gs://{CUR_BUCKET or RAW_BUCKET}/{CUR_PREFIX}`")

# -------------------- 탭 --------------------
tab_gen, tab_review, tab_export = st.tabs(["🧪 생성(초안)", "🗂️ 제출 리뷰", "📦 데이터 내보내기"])

# == 탭 1: 생성(초안) ==
with tab_gen:
    st.header("🧪 생성(초안)")
    prompt = st.text_area("학생의 상황을 자세히 입력:", height=180, key="admin_gen_prompt")
    col1, col2 = st.columns([1,1])
    with col1:
        gen_btn = st.button("AI 초안 생성", use_container_width=True)
    with col2:
        st.write("")

    if gen_btn:
        if not prompt.strip():
            st.warning("프롬프트를 입력하세요.")
        else:
            with st.spinner("생성 중..."):
                try:
                    ai_text = call_model(ENDPOINT, prompt)
                    st.session_state["admin_last_ai"] = ai_text
                    st.success("아래 초안을 확인하세요.")
                except Exception as e:
                    st.error("생성 실패"); st.exception(e)

    if st.session_state.get("admin_last_ai"):
        st.subheader("🤖 AI 초안")
        st.write(st.session_state["admin_last_ai"])

        st.subheader("✍️ 최종 승인본 작성 후 즉시 curated 저장")
        approved = st.text_area("최종 피드백", value=st.session_state["admin_last_ai"], height=240)
        if st.button("✅ 승인 저장(새 항목)", type="primary"):
            try:
                ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                day = datetime.utcnow().strftime("%Y-%m-%d")
                out = {
                    "timestamp": ts,
                    "prompt": prompt,
                    "ai_response": st.session_state["admin_last_ai"],
                    "approved_response": approved.strip(),
                    "approved_by": "admin",
                    "approved_at": ts,
                    "used_model": ENDPOINT,
                    "review_notes": "",
                }
                fname = f"{uuid.uuid4().hex[:10]}.json"
                key = f"{CUR_PREFIX}/{day}/{fname}"
                upload_json(CUR_BUCKET or RAW_BUCKET, key, out)
                st.success(f"curated 저장 완료: gs://{CUR_BUCKET or RAW_BUCKET}/{key}")
            except Exception as e:
                st.error("저장 실패"); st.exception(e)

# == 탭 2: 제출 리뷰 ==
with tab_review:
    st.header("🗂️ 제출 리뷰")
    if not RAW_BUCKET:
        st.info("raw 버킷이 설정되지 않았습니다. Secrets에 raw_bucket_name/raw_prefix를 추가하세요.")
    else:
        # ---- 필터 UI ----
        colf1, colf2, colf3, colf4 = st.columns([1,1,1,1])
        with colf1:
            start_date = st.date_input("시작일", value=date.today())
        with colf2:
            end_date = st.date_input("종료일", value=date.today())
        with colf3:
            keyword = st.text_input("키워드(프롬프트/응답/메모 검색)", "")
        with colf4:
            limit = st.number_input("최대 로드 수", min_value=50, max_value=2000, value=400, step=50)

        # ---- 키 목록/필터 ----
        all_keys = list_keys(RAW_BUCKET, RAW_PREFIX)
        cand_keys = filter_keys_by_date(all_keys, start_date, end_date)[: int(limit)]
        # 메타 로딩 (키워드 필터를 위해 일부 JSON 읽기)
        entries = load_entries(RAW_BUCKET, cand_keys)
        if keyword.strip():
            entries = [e for e in entries if contains_keyword(e, keyword)]
        st.caption(f"필터 결과: {len(entries)}건")

        # 목록 라벨 생성
        def label_of(e):
            d = e.get("timestamp") or key_date(e.get("_key",""))
            p = (e.get("prompt") or "").replace("\n"," ")
            if len(p) > 40: p = p[:40]+"…"
            return f"{d} | {p}"

        options = ["(선택)"] + [label_of(e) for e in entries]
        sel = st.selectbox("검토할 항목", options, index=0, key="review_select")

        if sel != "(선택)":
            idx = options.index(sel) - 1
            item = entries[idx]

            st.subheader("원본 제출")
            st.write("제출시각:", item.get("timestamp"))
            st.write("프롬프트:")
            st.code(item.get("prompt",""), language="text")
            st.write("AI 초안:")
            st.write(item.get("ai_response",""))

            st.subheader("✍️ 관리자 승인본(수정/보완하여 입력)")
            approved_text = st.text_area(
                "최종 피드백(학습데이터로 쓰일 답변)",
                value=item.get("approved_response", item.get("ai_response","")),
                height=260,
                key="approved_text_area",
            )
            colb1, colb2, colb3 = st.columns([1,1,1])
            with colb1:
                delete_after = st.checkbox("승인 후 raw 삭제", value=False)
            with colb2:
                notes = st.text_input("관리자 메모(선택)", value=item.get("review_notes",""))
            with colb3:
                save_btn = st.button("✅ 승인 저장", type="primary")

            if save_btn:
                try:
                    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                    out = {
                        **item,
                        "approved_response": approved_text.strip(),
                        "approved_by": "admin",
                        "approved_at": ts,
                        "review_notes": notes,
                        "source_raw_bucket": item.get("_bucket"),
                        "source_raw_key": item.get("_key"),
                    }
                    out_key = curated_key_from_raw(item.get("_key",""))
                    upload_json(CUR_BUCKET or RAW_BUCKET, out_key, out)
                    st.success(f"curated 저장 완료 → gs://{CUR_BUCKET or RAW_BUCKET}/{out_key}")
                    if delete_after:
                        try:
                            delete_object(item.get("_bucket"), item.get("_key"))
                            st.info("원본(raw) 삭제 완료")
                        except Exception as de:
                            st.warning(f"원본 삭제 실패: {de}")
                except Exception as se:
                    st.error("승인 저장 실패"); st.exception(se)

            # 이전/다음
            coln1, coln2 = st.columns([1,1])
            with coln1:
                if st.button("◀ 이전"):
                    if idx - 1 >= 0:
                        st.session_state["review_select"] = options[idx]  # 이전 라벨
                        st.rerun()
            with coln2:
                if st.button("다음 ▶"):
                    if idx + 1 < len(entries):
                        st.session_state["review_select"] = options[idx+2]  # 다음 라벨 (옵션에 '(선택)'이 있음)
                        st.rerun()

# == 탭 3: 데이터 내보내기 ==
with tab_export:
    st.header("📦 데이터 내보내기 (curated)")
    if not (CUR_BUCKET or RAW_BUCKET):
        st.info("curated 저장소가 설정되지 않았습니다.")
    else:
        colx1, colx2, colx3, colx4 = st.columns([1,1,1,1])
        with colx1:
            start_date2 = st.date_input("시작일", value=date.today(), key="exp_start")
        with colx2:
            end_date2 = st.date_input("종료일", value=date.today(), key="exp_end")
        with colx3:
            keyword2 = st.text_input("키워드(프롬프트/응답/메모 검색)", "", key="exp_kw")
        with colx4:
            limit2 = st.number_input("최대 로드 수", min_value=50, max_value=5000, value=1000, step=50, key="exp_lim")

        # curated 키 목록
        ckeys_all = list_keys(CUR_BUCKET or RAW_BUCKET, CUR_PREFIX)
        ckeys = filter_keys_by_date(ckeys_all, start_date2, end_date2)[: int(limit2)]
        centries = load_entries(CUR_BUCKET or RAW_BUCKET, ckeys)
        if keyword2.strip():
            centries = [e for e in centries if contains_keyword(e, keyword2)]
        st.caption(f"필터 결과: {len(centries)}건")

        if centries:
            df = to_dataframe(centries)
            st.dataframe(df.head(20), use_container_width=True)

            # CSV
            csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "⬇️ CSV 다운로드", data=csv_bytes, file_name="curated_export.csv", mime="text/csv"
            )

            # JSONL (Vertex 튜닝 포맷)
            lines = to_jsonl_lines(centries)
            jsonl_bytes = ("\n".join(lines)).encode("utf-8")
            st.download_button(
                "⬇️ JSONL 다운로드 (Vertex 튜닝용)", data=jsonl_bytes, file_name="curated_tuning.jsonl", mime="application/json"
            )
        else:
            st.info("내보낼 데이터가 없습니다. 날짜/키워드를 조정해 보세요.")
