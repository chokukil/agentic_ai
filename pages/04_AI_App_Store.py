import streamlit as st
import json
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

PROMPT_STORE_PATH = "store/prompt_store.json"
PROMPT_CONFIG_DIR = "prompt-config"
os.makedirs(PROMPT_CONFIG_DIR, exist_ok=True)
ASSETS_DIR = "assets"
AI_APP_STORE_PATH = "store/ai_app_store.json"

# 환경 변수에서 사번/이름 불러오기
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
EMP_NO = os.getenv("EMP_NO", "default_emp_no")
EMP_NAME = os.getenv("EMP_NAME", "default_emp_name")

# 세션 상태 초기화
if "aiapp_likes" not in st.session_state:
    st.session_state.aiapp_likes = {}
if "aiapp_like_clicked" not in st.session_state:
    st.session_state.aiapp_like_clicked = set()
if "aiapp_icon_upload" not in st.session_state:
    st.session_state.aiapp_icon_upload = None
if "aiapp_new_name" not in st.session_state:
    st.session_state.aiapp_new_name = ""
if "aiapp_new_desc" not in st.session_state:
    st.session_state.aiapp_new_desc = ""
if "aiapp_new_error" not in st.session_state:
    st.session_state.aiapp_new_error = ""
if "aiapp_icon_upload_key" not in st.session_state:
    st.session_state.aiapp_icon_upload_key = 0
if "aiapp_new_url" not in st.session_state:
    st.session_state.aiapp_new_url = ""
if "aiapp_new_prompt" not in st.session_state:
    st.session_state.aiapp_new_prompt = ""
if "aiapp_new_hash_tags" not in st.session_state:
    st.session_state.aiapp_new_hash_tags = ""

# 데이터 로드 (ai_app_store.json 기반)
def load_ai_app_store():
    if not os.path.exists(AI_APP_STORE_PATH) or os.path.getsize(AI_APP_STORE_PATH) == 0:
        return {"auto": [], "user": []}
    try:
        with open(AI_APP_STORE_PATH, encoding="utf-8") as f:
            data = json.load(f)
            result = {"auto": [], "user": []}
            for section in data.get("AIAppStore", []):
                t = section.get("type")
                if t in result:
                    result[t].extend(section.get("apps", []))
            return result
    except Exception as e:
        st.exception(e)
        return {"auto": [], "user": []}

apps_by_type = load_ai_app_store()

st.title("AI App Store")
st.markdown("AI App을 카드로 보고, 좋아요/HyThanks/시스템 이동/등록 기능을 사용할 수 있습니다.")

# 카드 UI 함수 (title 기반)
def app_card_v2(app, key_prefix=""):
    with st.container():
        col_icon, col_main, col_button = st.columns([1, 4, 1])
        with col_icon:
            icon_name = app.get("icon_name", None)
            if icon_name:
                icon_path = os.path.join(ASSETS_DIR, "icons", icon_name)
                if os.path.exists(icon_path):
                    st.image(icon_path, width=64)
                else:
                    st.write(f"아이콘 파일 없음: {icon_name}")
            else:
                # 원형 배경 + 이니셜
                try:
                    import matplotlib.pyplot as plt
                    import numpy as np
                except ImportError as e:
                    st.error("matplotlib 또는 numpy가 설치되어 있지 않습니다.")
                    st.stop()
                fig, ax = plt.subplots(figsize=(1,1))
                ax.add_patch(plt.Circle((0.5,0.5),0.5,color="#6c63ff"))
                ax.text(0.5,0.5,app.get("title", "?")[0].upper(),color="white",fontsize=32,ha="center",va="center",weight="bold")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
        with col_main:
            st.subheader(app.get("title", "(제목없음)"))
            st.write(app.get("description", ""))
            if app.get("hash_tag"):
                st.write(" ".join([f"`{tag}`" for tag in app.get("hash_tag", [])]))
            likes = app.get("like", 0)
            app_key = app.get("title", "")
            prefix_str = f"{key_prefix}_" if key_prefix else ""
            like_id = f"{prefix_str}{app_key}"
            if like_id not in st.session_state.aiapp_likes:
                st.session_state.aiapp_likes[like_id] = likes

            # columns 대신 세로로 배치
            if st.button("👍", key=f"like_{like_id}", help="좋아요"):
                if like_id not in st.session_state.aiapp_like_clicked:
                    st.session_state.aiapp_likes[like_id] += 1
                    st.session_state.aiapp_like_clicked.add(like_id)
            st.write(f"{st.session_state.aiapp_likes[like_id]}")
            if st.button("HyThanks", key=f"hythanks_{like_id}"):
                st.info("기능 준비 중입니다.")
            st.write(app.get("EMP_NAME", "-"))
        with col_button:
            url = app.get("url", "#")
            st.markdown(f'<a href="{url}" target="_blank" style="text-decoration:none;"><button style="background:#eee;padding:6px 12px;border-radius:6px;border:none;">시스템 이동</button></a>', unsafe_allow_html=True)

# 2열 카테고리별 표기
col_left, col_right = st.columns(2)
with col_left:
    st.markdown("#### AI ON Powered Apps")
    st.markdown("AI ON Agent Builder로 제작된 App입니다.")
    for idx, app in enumerate(sorted(apps_by_type["auto"], key=lambda x: x.get("like", 0), reverse=True)):
        app_card_v2(app, key_prefix=f"auto_{idx}")
        st.markdown("---")
with col_right:
    st.markdown("#### AI Apps")
    st.markdown("AI 관련 시스템은 모두 여기에 등록할 수 있습니다.")
    for idx, app in enumerate(sorted(apps_by_type["user"], key=lambda x: x.get("like", 0), reverse=True)):
        app_card_v2(app, key_prefix=f"user_{idx}")
        st.markdown("---")

st.markdown("---")

# AI App 등록 폼
st.subheader("신규 AI App 등록")

col1, col2 = st.columns([2,1])
with col1:
    st.session_state.aiapp_new_name = st.text_input("App 이름", value=st.session_state.aiapp_new_name, key="aiapp_new_name_input")
    st.session_state.aiapp_new_desc = st.text_input("App 설명", value=st.session_state.aiapp_new_desc, key="aiapp_new_desc_input")
    st.session_state.aiapp_new_url = st.text_input("App URL", value=st.session_state.aiapp_new_url, key="aiapp_new_url_input")
    st.session_state.aiapp_new_hash_tags = st.text_input("Hash Tags (comma-separated)", value=st.session_state.aiapp_new_hash_tags, key="aiapp_hash_tags_input")
with col2:
    icon_file = st.file_uploader(
        "App 아이콘 (선택, PNG/JPG)", 
        type=["png","jpg","jpeg"], 
        key=f"aiapp_icon_upload_{st.session_state.aiapp_icon_upload_key}"
    )
    if icon_file:
        st.session_state.aiapp_icon_upload = icon_file

def save_ai_app_store(apps_by_type):
    """
    Save the current apps_by_type dict to AI_APP_STORE_PATH in the correct format.
    """
    data = {"AIAppStore": []}
    for t in ["auto", "user"]:
        data["AIAppStore"].append({
            "type": t,
            "apps": apps_by_type.get(t, [])
        })
    with open(AI_APP_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if st.button("등록", key="aiapp_register_btn"):
    name = st.session_state.aiapp_new_name.strip()
    desc = st.session_state.aiapp_new_desc.strip()
    url = st.session_state.aiapp_new_url.strip()
    prompt = st.session_state.aiapp_new_prompt.strip()
    hash_tags_raw = st.session_state.aiapp_new_hash_tags.strip()
    hash_tags = [tag.strip() for tag in hash_tags_raw.split(",") if tag.strip()]
    icon_file = st.session_state.aiapp_icon_upload
    if not name or not desc:
        st.session_state.aiapp_new_error = "이름과 설명을 모두 입력하세요."
    elif name in [app.get("title", "") for app in apps_by_type["auto"]] or name in [app.get("title", "") for app in apps_by_type["user"]]:
        st.session_state.aiapp_new_error = f"이미 존재하는 App 이름입니다. 다른 이름을 입력하세요. (중복: {name})"
    else:
        # 아이콘 저장
        icon_path = None
        if icon_file:
            icon_ext = Path(icon_file.name).suffix
            icon_name = f"aiapp_{name}{icon_ext}"
            icon_save_path = os.path.join(ASSETS_DIR, "icons", icon_name)
            with open(icon_save_path, "wb") as f:
                f.write(icon_file.read())
            icon_path = icon_name
        # 등록
        apps_by_type["user"].append({
            "title": name,
            "description": desc,
            "EMP_NO": EMP_NO,
            "EMP_NAME": EMP_NAME,
            "like": 0,
            "icon_name": icon_path,
            "prompt": prompt,
            "url": url,
            "hash_tag": hash_tags,
            "tools": "",
        })
        save_ai_app_store(apps_by_type)
        st.success("신규 AI App이 등록되었습니다! 새로고침 후 확인하세요.")
        st.session_state.aiapp_new_name = ""
        st.session_state.aiapp_new_desc = ""
        st.session_state.aiapp_icon_upload = None
        st.session_state.aiapp_new_error = ""
        st.session_state.aiapp_icon_upload_key += 1

if st.session_state.aiapp_new_error:
    st.error(st.session_state.aiapp_new_error)

