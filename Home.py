# main.py
import streamlit as st
import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# — 기존 st.experimental_get_query_params() 대체 —
agent = st.query_params.get("agent")   # URL 파라미터 중 'agent'를 바로 꺼냄

# Load agent configurations dynamically from store/ai_app_store.json
config_path = Path(__file__).parent / "store" / "ai_app_store.json"
with open(config_path, encoding="utf-8") as f:
    store_data = json.load(f)

# Extract apps of type 'auto'
auto_apps = next((entry.get("apps", []) for entry in store_data.get("AIAppStore", []) if entry.get("type") == "auto"), [])

# Build mapping from agent key to app config
agents = {}
for app in auto_apps:
    parsed = urlparse(app.get("url", ""))
    params = parse_qs(parsed.query)
    key = params.get("agent", [""])[0]
    if key:
        agents[key] = app

if not agent:
    # 홈 화면
    st.title("🤖 AI Agent Hub")
    st.write("원하는 에이전트를 선택하세요.")
    
    # Display dynamic agent list
    for key, app in agents.items():
        title = app.get("title", key)
        # 링크 클릭 시 URL이 ?agent=<key> 으로 바뀜
        st.markdown(f"- [{title}](?agent={key})")
else:
    # agent 파라미터에 따라 동적 분기
    if agent in agents:
        app = agents[agent]
        st.header(app.get("title", agent))
        # TODO: 여기에 {app.get("title", agent)} 에이전트 로직을 구현하세요.
    else:
        st.error(f":warning: 알 수 없는 에이전트: `{agent}`")
