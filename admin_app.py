# admin_app.py — 관리자용(초안 생성, 제출 리뷰, CSV/JSONL 내보내기)
# -----------------------------------------------------------
# ✅ Streamlit Secrets (Settings → Secrets)
# project_id = "feedback-ai-prototype-ver05"
# location   = "us-central1"
# tuned_model_name = "projects/feedback-ai-prototype-ver05/locations/us-central1/tunedModels/2731304531139756032"
#
# raw_bucket_name = "feedback-proto-ai-raw"
# raw_prefix      = "raw_submissions"
# cur_bucket_name = "feedback-proto-ai-raw"     # 별도 버킷 쓰면 거기로 변경
# cur_prefix      = "curated"
#
# [gcp_service_account]
# ...서비스계정 JSON 원문 전체...
# -----------------------------------------------------------

from __future__ import annotations
import io
import json
import uuid
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Tuple

import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import storage

import vertexai
from vertexai.generative_models import GenerativeModel

# ---------------- 기본/Secrets ----------------
st.set_page_config(page_title="🐸 개구리 학습 피드백 (Admin)", page_icon="🛠️", layout="wide")

PROJECT_ID = st.secrets.get("project_id")
LOCATION   = st.secrets.get("location", "us-central1")
TUNED_NAME = (st.secrets.get("tuned_model_name") or "").strip()

RAW_BUCKET = st.secrets.get("raw_bucket_name")
RAW_PREFIX = (st.secrets.get("raw_prefix") or "raw_submissions").strip().strip("/")

CUR_BUCKET = st.secrets.get("cur_bucket_name", RAW_BUCKET)
CUR_PREFIX = (st.secrets.get("cur_prefix") or "curated").strip().strip("/")

if not (PROJECT_ID and LOCATION and TUNED_NAME and RAW_BUCKET and CUR_BUCKET):
    st.error("Secrets 설정이 부족합니다. project_id, location, tuned_model_name, raw/cur 버킷+프리픽스를 확인하세요.")
    st.stop()

# ---------------- 인증/클라이언트 ----------------
try:
    credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
except Exception as e:
    st.error("Secrets의 [gcp_service_account] JSON을 확인하세요.\n" + repr(e))
    st.stop()

vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
storage_client = storage.Client(project=PROJECT_ID, credentials=credentials)

# ---------------- 모델 호출 ----------------
def _gen_cfg() -> Dict[str, Any]:
    return {
        "max_output_tokens": 2048,     # 필요시 4096
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "response_mime_type": "text/plain",
    }

def call_model(prompt: str) -> Tuple[str, Dict[str, Any]]:
    """튜닝모델 스트리밍 → 동기 폴백"""
    meta: Dict[str, Any] = {"route": []}

    # 1) 스트리밍
    try:
        gm = GenerativeModel(TUNED_NAME)
        parts: List[str] = []
        for chunk in gm.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            generation_config=_gen_cfg(),
            stream=True,
        ):
            t = getattr(chunk, "text", None)
            if t:
                parts.append(t)
        text = "".join(parts).strip()
        meta["route"].append({"name": "tuned-stream", "ok": bool(text)})
        if text:
            return text, meta
    except Exception as e:
        meta["route"].append({"name": "tuned-stream", "error": repr(e)})

    # 2) 동기
    try:
        gm = GenerativeModel(TUNED_NAME)
        r = gm.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            generation_config=_gen_cfg(),
        )
        text = (getattr(r, "text", "") or "").strip()
        meta["route"].append({"name": "tuned-sync", "ok": bool(text)})
        return text, meta
    except Exception as e:
        meta["route"].append({"name": "tuned-sync", "error": repr(e)})
        return "", meta

# ---------------- GCS 유틸 ----------------
def gcs_upload_json(bucket: str, key: str, obj: Dict[str, Any]):
    b = storage_client.bucket(bucket).blob(key)
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    b.cache_control = "no-cache"
    b.upload_from_file(io.BytesIO(data), size=len(data), content_type="application/json")

def gcs_read_json(bucket: str, key: str) -> Dict[str, Any]:
    return json.loads(storage_client.bucket(bucket).blob(key).download_as_bytes())

def gcs_delete(bucket: str, key: str):
    storage_client.bucket(bucket).blob(key).delete()

@st.cache_data(ttl=60)
def list_keys(bucket: str, prefix: str) -> List[str]:
    try:
        blobs = storage_client.list_blobs(bucket, prefix=f"{prefix}/")
        keys = [b.name for b in blobs if b.name.endswith(".json")]
        keys.sort(reverse=True)
        return keys
    except Exception as e:
        st.warning(f"키 목록 로드 실패: {e}")
        return []

def key_date(key: str) -> str:
    parts = key.split("/")
    return parts[1] if len(parts) >= 3 else ""

def filter_keys_by_date(keys: List[str], start: date, end: date) -> List[str]:
    s, e = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    return [k for k in keys if (d:=key_date(k)) and (s <= d <= e)]

def load_entries(bucket: str, keys: List[str]) -> List[Dict[str, Any]]:
    out = []
    for k in keys:
        try:
            d = gcs_read_json(bucket, k)
            d["_bucket"] = bucket
            d["_key"] = k
            out.append(d)
        except Exception:
            pass
    return out

def curated_key_from_raw(raw_key: str) -> str:
    parts = raw_key.split("/", 2)
    if len(parts) >= 3:
        return f"{CUR_PREFIX}/{parts[1]}/{parts[2]}"
    day = datetime.utcnow().strftime("%Y-%m-%d")
    return f"{CUR_PREFIX}/{day}/{uuid.uuid4().hex[:10]}.json"

def contains_kw(e: Dict[str, Any], kw: str) -> bool:
    if not kw:
        return True
    kw = kw.lower()
    for f in ("prompt", "ai_response", "approved_response", "review_notes"):
        if kw in (e.get(f, "") or "").lower():
            return True
    return False

def to_jsonl_lines(entries: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for e in entries:
        prompt = (e.get("prompt") or "").strip()
        out = (e.get("approved_response") or e.get("ai_response") or "").strip()
        if not prompt or not out:
            continue
        lines.append(json.dumps({"contents":[
            {"role":"user","parts":[{"text":prompt}]},
            {"role":"model","parts":[{"text":out}]},
        ]}, ensure_ascii=False))
    return lines

def to_dataframe(entries: List[Dict[str, Any]]) -> pd.DataFrame:
    cols = ["timestamp","prompt","ai_response","approved_response","approved_by","approved_at",
            "review_notes","used_model","source_raw_bucket","source_raw_key","_bucket","_key"]
    rows = []
    for e in entries:
        rows.append({c: e.get(c) for c in cols})
    return pd.DataFrame(rows, columns=cols)

# ---------------- 사이드바 ----------------
with st.sidebar:
    st.markdown("### 환경 정보")
    st.write(f"Project: `{PROJECT_ID}`")
    st.write(f"Location: `{LOCATION}`")
    st.write(f"Tuned model:\n`{TUNED_NAME}`")
    st.write("---")
    st.write(f"raw: `gs://{RAW_BUCKET}/{RAW_PREFIX}`")
    st.write(f"curated: `gs://{CUR_BUCKET}/{CUR_PREFIX}`")

# ---------------- 탭 구성 ----------------
tab_gen, tab_review, tab_export = st.tabs(["🧪 생성(초안)", "🗂️ 제출 리뷰", "📦 데이터 내보내기"])

# === 탭 1: 생성(초안) ===
with tab_gen:
    st.header("🧪 생성(초안)")
    prompt = st.text_area("학생의 상황을 자세히 입력:", height=180, key="admin_gen_prompt")

    if st.button("AI 초안 생성", use_container_width=True, key="gen_btn"):
        if not prompt.strip():
            st.warning("프롬프트를 입력하세요.")
        else:
            with st.spinner("생성 중..."):
                text, meta = call_model(prompt)
                if text:
                    st.session_state["admin_last_ai"] = text
                    st.success("초안 생성 완료")
                else:
                    st.session_state["admin_last_ai"] = ""
                    st.warning("모델이 빈 응답을 반환했습니다.")
                    with st.expander("디버그"):
                        st.json(meta)

    if st.session_state.get("admin_last_ai"):
        st.subheader("🤖 AI 초안")
        st.text_area("AI 초안 출력", st.session_state["admin_last_ai"], height=280, key="gen_output")

        st.subheader("✍️ 최종 승인본 → curated 저장")
        approved = st.text_area("최종 피드백", value=st.session_state["admin_last_ai"], height=260, key="gen_approved")
        if st.button("✅ 승인 저장(새 항목)", type="primary", key="gen_save_btn"):
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
                    "review_notes": "",
                    "used_model": TUNED_NAME,
                    "source_raw_bucket": None,
                    "source_raw_key": None,
                }
                key = f"{CUR_PREFIX}/{day}/{uuid.uuid4().hex[:10]}.json"
                gcs_upload_json(CUR_BUCKET, key, out)
                st.success(f"curated 저장 완료: gs://{CUR_BUCKET}/{key}")
            except Exception as e:
                st.error("저장 실패"); st.exception(e)

# === 탭 2: 제출 리뷰 ===
with tab_review:
    st.header("🗂️ 제출 리뷰")
    today = date.today()
    c1,c2,c3,c4 = st.columns([1,1,1,1])
    with c1:
        start_d = st.date_input("시작일", value=today - timedelta(days=7), key="review_start_date")
    with c2:
        end_d   = st.date_input("종료일", value=today, key="review_end_date")
    with c3:
        kw = st.text_input("키워드(프롬프트/응답/메모 검색)", "", key="review_kw")
    with c4:
        limit = st.number_input("최대 로드 수", min_value=50, max_value=3000, value=600, step=50, key="review_limit")

    all_keys = list_keys(RAW_BUCKET, RAW_PREFIX)
    keys = filter_keys_by_date(all_keys, start_d, end_d)[: int(limit)]
    entries = load_entries(RAW_BUCKET, keys)
    if kw.strip():
        entries = [e for e in entries if contains_kw(e, kw)]
    st.caption(f"필터 결과: {len(entries)}건")

    def label_of(e):
        d = e.get("timestamp") or key_date(e.get("_key",""))
        p = (e.get("prompt") or "").replace("\n"," ")
        return f"{d} | {p[:40]}{'…' if len(p)>40 else ''}"

    options = ["(선택)"] + [label_of(e) for e in entries]
    sel = st.selectbox("검토할 항목", options, index=0, key="review_select")

    if sel != "(선택)":
        idx = options.index(sel) - 1
        item = entries[idx]

        st.subheader("원본 제출")
        st.write("제출시각:", item.get("timestamp"))
        st.write("프롬프트:"); st.code(item.get("prompt",""))
        st.write("AI 초안:");  st.text_area("원본 AI 초안", item.get("ai_response",""), height=220, key="review_ai_text")

        st.subheader("✍️ 승인본(수정/보완)")
        approved_text = st.text_area(
            "최종 피드백",
            value=item.get("approved_response", item.get("ai_response","")),
            height=260,
            key="review_approved_text"
        )
        cba, cbb, cbc = st.columns([1,1,1])
        with cba:
            delete_after = st.checkbox("승인 후 raw 삭제", value=False, key="review_delete_after")
        with cbb:
            notes = st.text_input("관리자 메모(선택)", value=item.get("review_notes",""), key="review_notes")
        with cbc:
            ok = st.button("✅ 승인 저장", type="primary", key="review_save_btn")

        if ok:
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
                gcs_upload_json(CUR_BUCKET, out_key, out)
                st.success(f"curated 저장 완료 → gs://{CUR_BUCKET}/{out_key}")
                if delete_after:
                    try:
                        gcs_delete(item.get("_bucket"), item.get("_key"))
                        st.info("원본(raw) 삭제 완료")
                    except Exception as de:
                        st.warning(f"원본 삭제 실패: {de}")
            except Exception as e:
                st.error("승인 저장 실패"); st.exception(e)

        prev, nxt = st.columns([1,1])
        with prev:
            if st.button("◀ 이전", key="review_prev_btn"):
                if idx-1 >= 0:
                    st.session_state["review_select"] = options[idx]
                    st.rerun()
        with nxt:
            if st.button("다음 ▶", key="review_next_btn"):
                if idx+1 < len(entries):
                    st.session_state["review_select"] = options[idx+2]
                    st.rerun()

# === 탭 3: 데이터 내보내기 ===
with tab_export:
    st.header("📦 데이터 내보내기 (curated)")
    c1,c2,c3,c4 = st.columns([1,1,1,1])
    with c1:
        s2 = st.date_input("시작일", value=date.today()-timedelta(days=30), key="export_start_date")
    with c2:
        e2 = st.date_input("종료일", value=date.today(), key="export_end_date")
    with c3:
        kw2 = st.text_input("키워드", "", key="export_kw")
    with c4:
        lim2 = st.number_input("최대 로드 수", min_value=50, max_value=5000, value=1500, step=50, key="export_limit")

    ckeys_all = list_keys(CUR_BUCKET, CUR_PREFIX)
    ckeys = filter_keys_by_date(ckeys_all, s2, e2)[: int(lim2)]
    centries = load_entries(CUR_BUCKET, ckeys)
    if kw2.strip():
        centries = [e for e in centries if contains_kw(e, kw2)]
    st.caption(f"필터 결과: {len(centries)}건")

    if centries:
        df = to_dataframe(centries)
        st.dataframe(df.head(30), use_container_width=True, key="export_df")

        csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("⬇️ CSV 다운로드", data=csv_bytes, file_name="curated_export.csv", mime="text/csv", key="export_csv")

        jsonl_bytes = ("\n".join(to_jsonl_lines(centries))).encode("utf-8")
        st.download_button("⬇️ JSONL 다운로드 (Vertex 튜닝용)", data=jsonl_bytes, file_name="curated_tuning.jsonl", mime="application/json", key="export_jsonl")
    else:
        st.info("내보낼 데이터가 없습니다. 날짜/키워드를 조정해 보세요.")
