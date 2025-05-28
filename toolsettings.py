import streamlit as st
import glob
import json
import os
from pathlib import Path
from copy import deepcopy

st.set_page_config(
    page_title="Tool 설정",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define the mcp-config directory
MCP_CONFIG_DIR = "mcp-config"
# Create directory if it doesn't exist
os.makedirs(MCP_CONFIG_DIR, exist_ok=True)

# 사이드바 최상단에 저자 정보 추가 (다른 사이드바 요소보다 먼저 배치)
st.sidebar.markdown("### 🤖 AI ON 💡")

st.sidebar.markdown("---")


# --- Sidebar for File Selection & Save ---
with st.sidebar:
    st.header("📂 설정 파일 선택 & 저장")
    # JSON 파일 목록
    json_paths = glob.glob(f"{MCP_CONFIG_DIR}/*.json")
    # If no JSON files found, add a default mcp_config.json option
    if not json_paths and not os.path.exists(f"{MCP_CONFIG_DIR}/mcp_config.json"):
        default_config = {"mcpServers": {}}
        with open(f"{MCP_CONFIG_DIR}/mcp_config.json", "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        json_paths = [f"{MCP_CONFIG_DIR}/mcp_config.json"]

    tools_list = [{"name": Path(p).stem, "path": p} for p in json_paths]
    selected_name = st.selectbox("설정 파일 선택", [t["name"] for t in tools_list])

    # Load 설정
    if st.button("📥 선택된 파일 Load", key="load", use_container_width=True):
        selected = next(t for t in tools_list if t["name"] == selected_name)
        with open(selected["path"], encoding="utf-8") as f:
            st.session_state.tool_config = json.load(f)
        st.session_state.file_path = selected["path"]
        st.session_state.loaded = True
        st.success(f"Loaded: {selected_name}.json")

    # Save 변경사항
    if st.session_state.get("loaded", False):
        if st.button("💾 저장", key="save", use_container_width=True):
            with open(st.session_state.file_path, "w", encoding="utf-8") as f:
                json.dump(st.session_state.tool_config, f, indent=2, ensure_ascii=False)
            st.session_state.saved_msg = (
                f"저장 완료: {Path(st.session_state.file_path).name}"
            )
            st.rerun()

# --- Main Area ---
st.title("🔧 MCP Agents Tool 설정")

if not st.session_state.get("loaded", False):
    st.info("사이드바에서 JSON 파일을 로드하세요.")
else:
    # 탭 구성: 목록, 추가, JSON 미리보기, Cursor AI, Claude Desktop
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📝 Tool 목록",
            "➕ 도구 추가",
            "🔍 JSON 미리보기",
            "💻 Cursor AI",
            "🤖 Claude Desktop",
        ]
    )

    # Tab1: Tool List
    with tab1:
        mcp = st.session_state.tool_config.get("mcpServers", {})
        if not mcp:
            st.warning("등록된 도구가 없습니다.")
        else:
            for name in list(mcp.keys()):
                col1, col2 = st.columns([9, 1])
                with col1:
                    st.write(f"• **{name}**")
                with col2:
                    if st.button("삭제", key=f"del_{name}"):
                        del st.session_state.tool_config["mcpServers"][name]
                        st.success(f"도구 '{name}' 삭제됨")
                        st.rerun()

    # Tab2: Add Tool
    with tab2:
        st.markdown("🔍 [Smithery 바로가기](https://smithery.ai/)")
        hint = """{
  "mcpServers": {
    "perplexity-search": {
      "command": "npx",
      "args": [
        "-y",
        "@smithery/cli@latest",
        "run",
        "@arjunkmrm/perplexity-search",
        "--key",
        "SMITHERY_API_KEY"
      ]
    }
  }
}
"""
        new_tool_text = st.text_area("도구 JSON 입력", hint, height=260)
        if st.button("추가", key="add_tool"):
            text = new_tool_text.strip()
            try:
                new_tool = json.loads(text)
            except json.JSONDecodeError:
                try:
                    new_tool = json.loads(f"{{{text}}}")
                except json.JSONDecodeError as e:
                    st.error(f"JSON 파싱 오류: {e}")
                    new_tool = None
            if new_tool is not None:
                if "mcpServers" in new_tool and isinstance(
                    new_tool["mcpServers"], dict
                ):
                    tools_data = new_tool["mcpServers"]
                else:
                    tools_data = new_tool
                for name, cfg in tools_data.items():
                    if "transport" not in cfg:
                        cfg["transport"] = "sse" if "url" in cfg else "stdio"
                    st.session_state.tool_config.setdefault("mcpServers", {})[
                        name
                    ] = cfg
                added = ", ".join(tools_data.keys())
                st.success(f"도구 '{added}' 추가됨")
                st.rerun()

    # Tab3: JSON Preview
    with tab3:
        st.code(
            json.dumps(st.session_state.tool_config, indent=2, ensure_ascii=False),
            language="json",
        )

    # Tab4: Cursor AI JSON Preview without transport
    with tab4:
        preview = deepcopy(st.session_state.tool_config)
        servers = preview.get("mcpServers", {})
        for cfg in servers.values():
            if isinstance(cfg, dict) and "transport" in cfg:
                del cfg["transport"]
        st.code(json.dumps(preview, indent=2, ensure_ascii=False), language="json")

    # Tab5: Claude Desktop JSON Preview without transport and URL
    with tab5:
        preview_cd = deepcopy(st.session_state.tool_config)
        servers_cd = preview_cd.get("mcpServers", {})
        # URL 파라미터가 있는 엔트리 확인 및 제거
        invalid = [
            name
            for name, cfg in servers_cd.items()
            if isinstance(cfg, dict) and "url" in cfg
        ]
        if invalid:
            st.error(
                f"Claude Desktop에서 지원하지 않는 'url' 파라미터가 포함되어 다음 도구를 제외했습니다: {', '.join(invalid)}"
            )
            for name in invalid:
                del servers_cd[name]
        # transport 제거
        for cfg in servers_cd.values():
            if isinstance(cfg, dict) and "transport" in cfg:
                del cfg["transport"]
        st.code(json.dumps(preview_cd, indent=2, ensure_ascii=False), language="json")

# 하단 저장 메시지 출력
with st.sidebar:
    if st.session_state.get("saved_msg"):
        st.success(st.session_state.pop("saved_msg"))
