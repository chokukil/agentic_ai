import streamlit as st
import json
import os
from dotenv import load_dotenv, find_dotenv

PROMPT_STORE_PATH = "store/prompt_store.json"
PROMPT_CONFIG_DIR = "prompt-config"
os.makedirs(PROMPT_CONFIG_DIR, exist_ok=True)

# 환경 변수에서 사번/이름 불러오기
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
EMP_NO = os.getenv("EMP_NO", "default_emp_no")
EMP_NAME = os.getenv("EMP_NAME", "default_emp_name")

# Prompt Store 데이터 로드

def load_prompt_store():
    if not os.path.exists(PROMPT_STORE_PATH) or os.path.getsize(PROMPT_STORE_PATH) == 0:
        return {}
    with open(PROMPT_STORE_PATH, encoding="utf-8") as f:
        data = json.load(f)
        return data.get("prompts", {})


# 세션 상태 초기화
if "prompt_cart" not in st.session_state:
    st.session_state.prompt_cart = []
    # 'AI ON Agent Default'가 prompt store에 있으면 기본적으로 추가
    prompts = load_prompt_store()
    if "AI ON Agent Default" in prompts and "AI ON Agent Default" not in st.session_state.prompt_cart:
        st.session_state.prompt_cart.append("AI ON Agent Default")



prompts = load_prompt_store()

st.title("Prompt Store")
st.markdown("Prompt를 장바구니에 담아 Checkout 하세요!")

# Prompt 카드 UI
def prompt_card(title, info):
    with st.container():
        st.subheader(f"{title}")
        prompt_text = info.get('prompt', '-')
        # 미리보기: 앞 100자만 보여주고, 더 길면 ... 표시
        preview_len = 200
        preview_text = prompt_text[:preview_len] + ("..." if len(prompt_text) > preview_len else "")
        st.write(f"**미리보기:** {preview_text}")
        if len(prompt_text) > preview_len:
            with st.expander("Prompt 전체 펼치기/접기", expanded=False):
                st.write(prompt_text)
        # 버튼 4개: 장바구니, 삭제, HyThanks, EMP_NAME
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1, 1, 1.5, 1.2])
        with btn_col1:
            if title in st.session_state.prompt_cart:
                if st.button(f"❌", key=f"remove_{title}"):
                    st.session_state.prompt_cart.remove(title)
                    st.rerun()
            else:
                if st.button(f"🛒", key=f"add_{title}"):
                    st.session_state.prompt_cart.append(title)
                    st.rerun()
        with btn_col2:
            # 삭제 버튼: 본인(EMP_NO)만
            if info.get("EMP_NO") == EMP_NO:
                if st.button("삭제", key=f"delete_{title}"):
                    # 삭제 처리: prompt_store.json에서 삭제
                    try:
                        if os.path.exists(PROMPT_STORE_PATH) and os.path.getsize(PROMPT_STORE_PATH) > 0:
                            with open(PROMPT_STORE_PATH, encoding="utf-8") as f:
                                store_data = json.load(f)
                            if "prompts" in store_data and title in store_data["prompts"]:
                                del store_data["prompts"][title]
                                # 장바구니에서도 제거
                                if title in st.session_state.prompt_cart:
                                    st.session_state.prompt_cart.remove(title)
                                with open(PROMPT_STORE_PATH, "w", encoding="utf-8") as f:
                                    json.dump(store_data, f, indent=2, ensure_ascii=False)
                                st.success(f"Prompt '{title}'가 삭제되었습니다.")
                                st.rerun()
                    except Exception as e:
                        st.error(f"삭제 실패: {e}")
        with btn_col3:
            if st.button("HyThanks", key=f"hythanks_{title}"):
                st.info("기능 준비 중입니다.")
        with btn_col4:
            if st.button(info.get("EMP_NAME", "-"), key=f"empname_{title}"):
                st.info("조직도 연계 준비중입니다.")

# 카드 그리드
num_columns = 2
cols = st.columns(num_columns)
prompt_items = list(prompts.items())
for idx, (title, info) in enumerate(prompt_items):
    with cols[idx % num_columns]:
        prompt_card(title, info)
        if idx < len(prompt_items) - num_columns or (len(prompt_items) % num_columns != 0 and idx >= len(prompt_items) - (len(prompt_items) % num_columns)):
            st.markdown("---")

st.markdown("---")

# 장바구니 영역 (사이드바)
with st.sidebar:
    st.header("🛒 장바구니")
    if not st.session_state.prompt_cart:
        st.info("장바구니가 비어 있습니다.")
    else:
        for title in st.session_state.prompt_cart:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(title)
            with col2:
                if st.button(f"❌", key=f"remove_cart_{title}"):
                    st.session_state.prompt_cart.remove(title)
                    st.rerun()

    # 장바구니 저장 (이름 입력 없이 EMP_NO로 저장)
    if st.button("장바구니 저장", key="save_cart_btn"):
        selected_prompts = {}
        for title in st.session_state.prompt_cart:
            prompt_info = prompts[title].copy()
            prompt_info["EMP_NO"] = EMP_NO
            prompt_info["EMP_NAME"] = EMP_NAME
            selected_prompts[title] = prompt_info
        save_path = os.path.join(PROMPT_CONFIG_DIR, f"{EMP_NO}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"prompts": selected_prompts}, f, indent=2, ensure_ascii=False)
        st.success(f"저장 완료. Agent Builder에서 확인하세요.")

# 신규 Prompt 등록
st.subheader("신규 Prompt 등록")

if "new_prompt_title" not in st.session_state:
    st.session_state.new_prompt_title = ""
if "new_prompt_text" not in st.session_state:
    st.session_state.new_prompt_text = ""
if "new_prompt_error" not in st.session_state:
    st.session_state.new_prompt_error = ""

st.session_state.new_prompt_title = st.text_input("Prompt 제목", value=st.session_state.new_prompt_title, key="new_prompt_title_input")
st.session_state.new_prompt_text = st.text_area("Prompt 내용", value=st.session_state.new_prompt_text, height=120, key="new_prompt_text_area")

if st.button("등록", key="register_prompt_btn"):
    title = st.session_state.new_prompt_title.strip()
    prompt_text = st.session_state.new_prompt_text.strip()
    if not title or not prompt_text:
        st.session_state.new_prompt_error = "제목과 내용을 모두 입력하세요."
    elif title in prompts:
        st.session_state.new_prompt_error = f"이미 존재하는 제목입니다. 다른 제목을 입력하세요. (중복: {title})"
    else:
        # 등록
        try:
            # 기존 데이터 불러오기
            if os.path.exists(PROMPT_STORE_PATH) and os.path.getsize(PROMPT_STORE_PATH) > 0:
                with open(PROMPT_STORE_PATH, encoding="utf-8") as f:
                    store_data = json.load(f)
                if "prompts" not in store_data:
                    store_data["prompts"] = {}
            else:
                store_data = {"prompts": {}}
            store_data["prompts"][title] = {"prompt": prompt_text, "EMP_NO": EMP_NO, "EMP_NAME": EMP_NAME}
            with open(PROMPT_STORE_PATH, "w", encoding="utf-8") as f:
                json.dump(store_data, f, indent=2, ensure_ascii=False)
            st.success("신규 Prompt가 등록되었습니다! 새로고침 후 확인하세요.")
            st.session_state.new_prompt_title = ""
            st.session_state.new_prompt_text = ""
            st.session_state.new_prompt_error = ""
        except Exception as e:
            st.session_state.new_prompt_error = f"등록 실패: {e}"

if st.session_state.new_prompt_error:
    st.error(st.session_state.new_prompt_error)
