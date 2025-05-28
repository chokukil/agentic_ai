import streamlit as st
import json
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# MCP Store 데이터 경로
MCP_STORE_PATH = "store/mcp_store.json"
MCP_CONFIG_DIR = "mcp-config"
os.makedirs(MCP_CONFIG_DIR, exist_ok=True)

# 환경 변수에서 사번/이름 불러오기
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
EMP_NO = os.getenv("EMP_NO", "default_emp_no")
EMP_NAME = os.getenv("EMP_NAME", "default_emp_name")

# 세션 상태 초기화
if "cart" not in st.session_state:
    st.session_state.cart = []
if "checkout_name" not in st.session_state:
    st.session_state.checkout_name = ""

# MCP Store 데이터 로드
def load_mcp_store():
    with open(MCP_STORE_PATH, encoding="utf-8") as f:
        return json.load(f)["mcpServers"]

mcp_servers = load_mcp_store()

st.title("MCP Store")
st.markdown("MCP 서버를 장바구니에 담아 원하는 조합으로 Checkout 하세요!")

# MCP 서버 카드 UI
def mcp_card(name, info):
    with st.container():
        st.subheader(f"{name}")
        st.write(f"**설명:** {info.get('description', '-')}")
        # JSON 보기 토글 버튼
        json_key = f"show_json_{name}"
        if json_key not in st.session_state:
            st.session_state[json_key] = False
        # 버튼을 가로로 배치 (장바구니, JSON, HyThanks, EMP_NAME)
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1, 1.2, 1.5, 1.2])
        with btn_col1:
            if name in st.session_state.cart:
                if st.button(f"❌", key=f"remove_{name}"):
                    st.session_state.cart.remove(name)
                    st.rerun()
            else:
                if st.button(f"🛒", key=f"add_{name}"):
                    st.session_state.cart.append(name)
                    st.rerun()
        with btn_col2:
            if st.button("JSON" if not st.session_state[json_key] else "숨기기", key=f"toggle_{name}"):
                st.session_state[json_key] = not st.session_state[json_key]
                st.rerun()
        with btn_col3:
            if st.button("HyThanks", key=f"hythanks_{name}"):
                st.info("기능 준비 중입니다.")
        with btn_col4:
            if st.button(EMP_NAME, key=f"empname_{name}"):
                st.info("조직도 연계 준비중입니다.")
        if st.session_state[json_key]:
            st.code(json.dumps(info, indent=2, ensure_ascii=False), language="json")

# 카드 그리드
num_columns = 2  # Number of columns
cols = st.columns(num_columns)
server_items = list(mcp_servers.items())
for idx, (name, info) in enumerate(server_items):
    with cols[idx % num_columns]:
        mcp_card(name, info)
        # 마지막 줄이 아니면 구분선 추가
        # 3개씩 한 줄이므로, 마지막 1~3개는 제외
        if idx < len(server_items) - num_columns or (len(server_items) % num_columns != 0 and idx >= len(server_items) - (len(server_items) % num_columns)):
            st.markdown("---")

st.markdown("---")

# 장바구니 영역 (이제 사이드바에 고정됨)
with st.sidebar:
    st.header("🛒 장바구니")
    if not st.session_state.cart:
        st.info("장바구니가 비어 있습니다.")
    else:
        for name in st.session_state.cart:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(name)
            with col2:
                if st.button(f"❌", key=f"remove_cart_{name}"):
                    st.session_state.cart.remove(name)
                    st.rerun()

    with st.form("checkout_form"):
        checkout_name = st.text_input("장바구니 이름을 지어주세요", value=st.session_state.checkout_name)
        submitted = st.form_submit_button("저장")
        if submitted:
            if not checkout_name.strip():
                st.warning("이름을 입력하세요.")
            else:
                # cart에 담긴 이름을 기반으로 mcp_servers에서 정보 추출
                selected_servers = {name: mcp_servers[name] for name in st.session_state.cart}
                save_path = os.path.join(MCP_CONFIG_DIR, f"{checkout_name.strip()}.json")
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump({"mcpServers": selected_servers}, f, indent=2, ensure_ascii=False)
                st.success(f" 저장 완료. Agent Builder에서 확인하세요.")
                st.session_state.checkout_name = checkout_name.strip()

# 신규 MCP 등록
st.subheader("신규 MCP 등록")

# 신규 MCP 등록 상태 초기화
if "new_mcp_json" not in st.session_state:
    st.session_state.new_mcp_json = ""
if "new_mcp_valid" not in st.session_state:
    st.session_state.new_mcp_valid = False
if "new_mcp_missing_desc" not in st.session_state:
    st.session_state.new_mcp_missing_desc = False
if "new_mcp_desc" not in st.session_state:
    st.session_state.new_mcp_desc = ""
if "new_mcp_parsed" not in st.session_state:
    st.session_state.new_mcp_parsed = None
if "new_mcp_error" not in st.session_state:
    st.session_state.new_mcp_error = ""

st.markdown("""
신규 MCP 서버 정보를 JSON 형태로 입력하세요.\n
예시:
```json
{
  "my-mcp": {
    "command": "npx",
    "args": ["-y", "@myorg/my-mcp"],
    "description": "설명 예시"
  }
}
```
""")

st.session_state.new_mcp_json = st.text_area(
    "MCP JSON 입력", value=st.session_state.new_mcp_json, height=180, key="new_mcp_json_area"
)

# 유효성 검사 함수
def validate_new_mcp(json_text):
    try:
        parsed = json.loads(json_text)
        if not isinstance(parsed, dict):
            return False, None, "최상위는 객체여야 합니다.", False
        # mcpServers 래핑 여부
        if "mcpServers" in parsed:
            servers = parsed["mcpServers"]
        else:
            servers = parsed
        if not isinstance(servers, dict) or not servers:
            return False, None, "mcpServers(혹은 최상위)가 비어있거나 객체가 아닙니다.", False
        missing_desc = False
        for name, info in servers.items():
            if not isinstance(info, dict):
                return False, None, f"{name}의 값이 객체가 아닙니다.", False
            if "description" not in info or not info["description"]:
                missing_desc = True
        return True, servers, "", missing_desc
    except Exception as e:
        return False, None, str(e), False

# 유효성 검사 버튼/등록 버튼 상태
btn_label = "유효성 검사"
btn_color = "secondary"
if st.session_state.new_mcp_valid and not st.session_state.new_mcp_missing_desc:
    btn_label = "등록"
    btn_color = "success"

col1, col2 = st.columns([2, 1])
with col1:
    if st.button(btn_label, key="validate_or_register", type="primary" if btn_color=="success" else "secondary"):
        valid, servers, err, missing_desc = validate_new_mcp(st.session_state.new_mcp_json)
        st.session_state.new_mcp_valid = valid
        st.session_state.new_mcp_error = err
        st.session_state.new_mcp_missing_desc = missing_desc
        st.session_state.new_mcp_parsed = servers
        if valid and not missing_desc and btn_label == "등록":
            # 실제 등록 처리
            try:
                # 기존 MCP 불러오기
                with open(MCP_STORE_PATH, encoding="utf-8") as f:
                    store_data = json.load(f)
                if "mcpServers" not in store_data:
                    store_data["mcpServers"] = {}
                # 덮어쓰기/추가 (EMP_NO, EMP_NAME 추가)
                for name, info in servers.items():
                    info["EMP_NO"] = EMP_NO
                    info["EMP_NAME"] = EMP_NAME
                    info["transport"] = "stdio"
                store_data["mcpServers"].update(servers)
                with open(MCP_STORE_PATH, "w", encoding="utf-8") as f:
                    json.dump(store_data, f, indent=2, ensure_ascii=False)
                st.success("신규 MCP가 등록되었습니다! 새로고침 후 확인하세요.")
                # 상태 초기화
                st.session_state.new_mcp_json = ""
                st.session_state.new_mcp_valid = False
                st.session_state.new_mcp_missing_desc = False
                st.session_state.new_mcp_desc = ""
                st.session_state.new_mcp_parsed = None
                st.session_state.new_mcp_error = ""
            except Exception as e:
                st.error(f"등록 실패: {e}")

with col2:
    if st.session_state.new_mcp_missing_desc:
        st.session_state.new_mcp_desc = st.text_input("description이 없는 MCP에 설명을 입력하세요", value=st.session_state.new_mcp_desc, key="desc_input")
        # description 추가 버튼
        if st.button("설명 추가", key="add_desc_btn"):
            # description이 없는 MCP에 설명 추가
            servers = st.session_state.new_mcp_parsed
            if servers:
                for name, info in servers.items():
                    if "description" not in info or not info["description"]:
                        info["description"] = st.session_state.new_mcp_desc
                # JSON 텍스트 갱신
                st.session_state.new_mcp_json = json.dumps(servers, indent=2, ensure_ascii=False)
                # 유효성 재검사
                valid, servers, err, missing_desc = validate_new_mcp(st.session_state.new_mcp_json)
                st.session_state.new_mcp_valid = valid
                st.session_state.new_mcp_error = err
                st.session_state.new_mcp_missing_desc = missing_desc
                st.session_state.new_mcp_parsed = servers
                if not missing_desc:
                    st.success("설명이 추가되었습니다. 다시 유효성 검사를 진행하세요.")

# 에러 메시지 출력
if st.session_state.new_mcp_error:
    st.error(f"오류: {st.session_state.new_mcp_error}")
