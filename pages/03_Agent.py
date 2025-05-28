import streamlit as st
import glob
import json
import os
from pathlib import Path
from copy import deepcopy
from dotenv import load_dotenv, find_dotenv
import asyncio
import nest_asyncio
import platform
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from utils import astream_graph, random_uuid
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
import requests
import sys
from contextlib import asynccontextmanager
from mcp.client import stdio as _stdio
import logging
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import ConfigDict
import yaml
from langchain_openai import ChatOpenAI
from urllib.parse import urlsplit, parse_qs
from utils import generate_followups, get_followup_llm

# Base directory for app icons
ASSETS_DIR = "assets"
URL_BASE = "http://localhost:2025/Agent?id="

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Log session state initialization
logging.debug('Initializing session state')

if platform.system() == "Windows":
    logging.debug(f"Using proactor: IocpProactor")
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# nest_asyncio 적용: 이미 실행 중인 이벤트 루프 내에서 중첩 호출 허용
nest_asyncio.apply()

# 전역 이벤트 루프 생성 및 재사용 (한번 생성한 후 계속 사용)
if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)


st.set_page_config(
    page_title="AI Agent Builder",
    layout="wide",
    initial_sidebar_state="expanded",
)


OUTPUT_TOKEN_INFO = {
    "o4-mini": {"max_tokens": 16000},
    "gpt-4o": {"max_tokens": 16000},
}

# Log function entry and exit
logging.debug('Entering function: initialize_session')
async def initialize_session(mcp_config=None):
    logging.debug('Initializing MCP session')
    with st.spinner("🔄 MCP 서버에 연결 중..."):
        await cleanup_mcp_client()
        logging.debug('MCP client cleaned up')

        # mcp_config이 None이거나 tool_config가 없는 경우 MCP 연결을 건너뜁니다.
        if mcp_config is None and (
            "tool_config" not in st.session_state or st.session_state.tool_config is None
        ):
            st.warning("⚠️ MCP 서버 연결을 건너뜁니다. 사이드바에서 MCP Tool을 선택해주세요.")
            st.session_state.tool_count = 0
            st.session_state.mcp_client = None
            st.session_state.session_initialized = True
            logging.debug('No tool configuration found, skipping MCP connection.')
            return True

        # mcp_config이 None이면 사이드바에서 로드된 tool_config 사용
        if mcp_config is None:
            mcp_config = st.session_state.tool_config

        # mcpServers 키가 있으면 해제
        connections = mcp_config.get("mcpServers", mcp_config)
        
        # Store connections for debugging
        st.session_state.last_mcp_connections = connections
        logging.debug(f"MCP connections configuration: {json.dumps(connections, indent=2)}")
        
        # MCP 서버 설정이 비어 있으면 건너뜁니다.
        if not connections:
            st.warning("⚠️ MCP 서버 설정이 비어 있습니다. MCP 연결을 건너뜁니다.")
            st.session_state.tool_count = 0
            st.session_state.mcp_client = None
            st.session_state.session_initialized = True
            logging.debug('MCP server configuration is empty, skipping connection.')
            return True

        # Initialize MCP client and connect to servers
        try:
            logging.debug("Creating MultiServerMCPClient with connections")
            client = MultiServerMCPClient(connections)
            
            logging.debug("Entering MCP client context")
            await client.__aenter__()
            logging.debug('MCP servers connected via context manager.')
            
            try:
                # Get and log available tools
                logging.debug("Retrieving tools from MCP client")
                tools = client.get_tools()
                tool_count = len(tools)
                st.session_state.tool_count = tool_count
                
                # Log individual tool details
                logging.debug(f"Retrieved {tool_count} tools from MCP client")
                for i, tool in enumerate(tools):
                    tool_name = getattr(tool, 'name', f"Tool_{i}")
                    logging.debug(f"Tool {i}: {tool_name}")
                    if hasattr(tool, 'args'):
                        logging.debug(f"Tool {i} args: {tool.args}")
                    if hasattr(tool, 'description'):
                        logging.debug(f"Tool {i} description: {tool.description}")
                
                st.session_state.mcp_client = client
            except Exception as e:
                logging.error(f"Error retrieving tools: {str(e)}")
                import traceback
                logging.error(f"Tool retrieval error details:\n{traceback.format_exc()}")
                st.error(f"MCP 도구를 가져오는 중 오류가 발생했습니다: {str(e)}")
                # Continue with empty tools list
                tools = []
                tool_count = 0
                st.session_state.tool_count = 0
            
            # Create React agent graph if tools are available
            if tool_count > 0:
                # Replace HTTPChatModel usage with ChatOpenAI
                load_dotenv()
                # Construct OpenAI API base URL (always host + /v1)
                raw_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com")
                # Remove any path, keep only scheme and netloc
                parsed = urlsplit(raw_base)
                openai_api_base = f"{parsed.scheme}://{parsed.netloc}/v1"
                openai_api_key = os.getenv("OPENAI_API_KEY", "")
                logging.debug(f"Creating ChatOpenAI with base_url: {openai_api_base}")
                model = ChatOpenAI(
                    model=st.session_state.selected_model,
                    temperature=st.session_state.temperature,
                    max_tokens=OUTPUT_TOKEN_INFO[st.session_state.selected_model]["max_tokens"],
                    api_key=openai_api_key,
                    base_url=openai_api_base,
                )
                try:
                    agent = create_react_agent(
                        model,
                        tools,
                        prompt=st.session_state.selected_prompt_text,
                        checkpointer=MemorySaver(),
                    )
                    st.session_state.agent = agent
                except Exception as e:
                    logging.error(f"Error creating ReAct agent: {str(e)}")
                    import traceback
                    logging.error(f"Agent creation error details:\n{traceback.format_exc()}")
                    st.error(f"에이전트 생성 중 오류가 발생했습니다: {str(e)}")
                    st.session_state.agent = None
                logging.debug('React agent creation complete')
            else:
                st.session_state.agent = None
                logging.warning('No tools available, React agent not created.')
            
            st.session_state.session_initialized = True
            return True
            
        except Exception as e:
            import traceback
            error_msg = f"Error initializing MCP client: {str(e)}"
            logging.error(f"{error_msg}\n{traceback.format_exc()}")
            st.error(error_msg)
            st.session_state.session_initialized = False
            return False

# Log function entry and exit
logging.debug('Entering function: cleanup_mcp_client')
async def cleanup_mcp_client():
    """
    기존 MCP 클라이언트를 안전하게 종료합니다.

    기존 클라이언트가 있는 경우 정상적으로 리소스를 해제합니다.
    """
    if "mcp_client" in st.session_state and st.session_state.mcp_client is not None:
        try:
            await st.session_state.mcp_client.__aexit__(None, None, None)
            st.session_state.mcp_client = None
        except Exception as e:
            import traceback
            logging.error(f"MCP client cleanup error: {str(e)}\n{traceback.format_exc()}")
    logging.debug('Exiting function: cleanup_mcp_client')


def print_message():
    """
    채팅 기록을 화면에 출력합니다.

    사용자와 어시스턴트의 메시지를 구분하여 화면에 표시하고,
    도구 호출 정보는 어시스턴트 메시지 컨테이너 내에 표시합니다.
    """
    i = 0
    while i < len(st.session_state.history):
        message = st.session_state.history[i]

        if message["role"] == "user":
            st.chat_message("user", avatar="🧑🏻").write(message["content"])
            i += 1
        elif message["role"] == "assistant":
            # 어시스턴트 메시지 컨테이너 생성
            with st.chat_message("assistant", avatar="🤖"):
                # 어시스턴트 메시지 내용 표시
                st.write(message["content"])

                # --- Followup 버튼 렌더링 ---
                followups = message.get("followups")
                if followups:
                    st.markdown("<div style='margin-top: 0.5em; margin-bottom: 0.5em; color: #888;'>후속 질문 제안:</div>", unsafe_allow_html=True)
                    btn_cols = st.columns(len(followups))
                    for idx, followup in enumerate(followups):
                        if btn_cols[idx].button(followup, key=f"followup_{i}_{idx}"):
                            st.session_state["user_query"] = followup
                            st.rerun()

                # 다음 메시지가 도구 호출 정보인지 확인
                if (
                    i + 1 < len(st.session_state.history)
                    and st.session_state.history[i + 1]["role"] == "assistant_tool"
                ):
                    # 도구 호출 정보를 동일한 컨테이너 내에 expander로 표시
                    with st.expander("🔧 도구 호출 정보", expanded=False):
                        st.write(st.session_state.history[i + 1]["content"])
                    i += 2  # 두 메시지를 함께 처리했으므로 2 증가
                else:
                    i += 1  # 일반 메시지만 처리했으므로 1 증가
        else:
            # assistant_tool 메시지는 위에서 처리되므로 건너뜀
            i += 1


def get_streaming_callback(text_placeholder, tool_placeholder):
    """
    스트리밍 콜백 함수를 생성합니다.

    이 함수는 LLM에서 생성되는 응답을 실시간으로 화면에 표시하기 위한 콜백 함수를 생성합니다.
    텍스트 응답과 도구 호출 정보를 각각 다른 영역에 표시합니다.

    매개변수:
        text_placeholder: 텍스트 응답을 표시할 Streamlit 컴포넌트
        tool_placeholder: 도구 호출 정보를 표시할 Streamlit 컴포넌트

    반환값:
        callback_func: 스트리밍 콜백 함수
        accumulated_text: 누적된 텍스트 응답을 저장하는 리스트
        accumulated_tool: 누적된 도구 호출 정보를 저장하는 리스트
    """
    accumulated_text = []
    accumulated_tool = []
    # Track tool call IDs to prevent duplicate pending calls
    seen_tool_call_ids = set()

    def callback_func(message: dict):
        nonlocal accumulated_text, accumulated_tool
        logging.debug(f"Streaming callback received message: {type(message)}")
        message_content = message.get("content", None)
        
        # Log message content for debugging
        if message_content:
            logging.debug(f"Message content type: {type(message_content)}")
            if hasattr(message_content, "content"):
                content_sample = str(message_content.content)[:100] + "..." if len(str(message_content.content)) > 100 else str(message_content.content)
                logging.debug(f"Content: {content_sample}")
        else:
            logging.debug("No message content found")
            
        # Handle complete AIMessage (non-chunked)
        if isinstance(message_content, AIMessage):
            logging.debug("Processing complete AIMessage")
            content = message_content.content
            if isinstance(content, str):
                accumulated_text.append(content)
                text_placeholder.write("".join(accumulated_text))
                logging.debug(f"Added complete AIMessage content to text: {content[:100]}...")
            
            # Check for tool calls in additional_kwargs
            if hasattr(message_content, "additional_kwargs") and "tool_calls" in message_content.additional_kwargs and message_content.additional_kwargs["tool_calls"]:
                tool_calls = message_content.additional_kwargs["tool_calls"]
                logging.debug("Found tool_calls in AIMessage additional_kwargs")
                for tool_call in tool_calls:
                    tool_id = tool_call.get("id")
                    tool_name = tool_call.get("function", {}).get("name", "")
                    raw_arguments = tool_call.get("function", {}).get("arguments", None)
                    if isinstance(raw_arguments, str):
                        try:
                            args = json.loads(raw_arguments)
                        except:
                            args = {"raw_arguments": raw_arguments}
                    else:
                        args = raw_arguments
                    # Accumulate formatted tool call info
                    call_info = {"name": tool_name, "args": args, "id": tool_id, "type": "tool_call"}
                    # Display only raw_arguments when available
                    if raw_arguments is not None:
                        if isinstance(raw_arguments, str):
                            raw_display = raw_arguments
                        else:
                            raw_display = json.dumps(raw_arguments, indent=2, ensure_ascii=False)
                        if tool_name:
                            if tool_name=='execute_python':
                                entry = f"\n**도구 호출: {tool_name}**\n\n{raw_display}\n"
                            else:
                                entry = f"\n**도구 호출: {tool_name}**\n\n{raw_display}\n"
                        else:
                            entry = raw_display
                            # entry = raw_display.replace("\n", "")
                        accumulated_tool.append(entry)
                    # Store tool call for later processing (only once per call)
                    if tool_id and tool_name and tool_id not in seen_tool_call_ids:
                        seen_tool_call_ids.add(tool_id)
                        if "pending_tool_calls" not in st.session_state:
                            st.session_state.pending_tool_calls = []
                        st.session_state.pending_tool_calls.append({
                            "id": tool_id,
                            "name": tool_name,
                            "arguments": args
                        })
                        logging.debug(f"Added pending tool call: {tool_name} (id: {tool_id})")
                tool_placeholder.write("".join(accumulated_tool))
            
            return None

        # Handle AIMessageChunk
        if isinstance(message_content, AIMessageChunk):
            logging.debug("Processing AIMessageChunk")
            content = message_content.content
            
            # Content is list (e.g., Claude format)
            if isinstance(content, list) and len(content) > 0:
                logging.debug(f"AIMessageChunk content is list of length {len(content)}")
                message_chunk = content[0]
                if message_chunk.get("type") == "text":
                    text = message_chunk.get("text", "")
                    accumulated_text.append(text)
                    text_placeholder.write("".join(accumulated_text))
                    logging.debug(f"Added text chunk: {text[:100]}...")
                elif message_chunk.get("type") == "tool_use":
                    logging.debug("Processing tool_use chunk")
                    # Handle partial JSON fragments if available
                    if "partial_json" in message_chunk:
                        partial = message_chunk["partial_json"]
                        try:
                            pretty = json.dumps(json.loads(partial), indent=2, ensure_ascii=False)
                        except Exception:
                            pretty = partial
                        entry = f"\n```json\n{pretty}\n```\n"
                    else:
                        # Fallback to full tool_call_chunks
                        chunks = getattr(message_content, "tool_call_chunks", None)
                        if chunks:
                            chunk = chunks[0]
                            entry = f"\n```json\n{str(chunk)}\n```\n"
                        else:
                            entry = ""
                    accumulated_tool.append(entry)
                    tool_placeholder.write("".join(accumulated_tool))
                # Skip non-text, non-tool_use chunks
            # Check for OpenAI style tool calls
            elif (
                hasattr(message_content, "tool_calls")
                and message_content.tool_calls
                and len(message_content.tool_calls) > 0
            ):
                logging.debug("Found tool_calls attribute in AIMessageChunk")
                tool_call_info = message_content.tool_calls[0]
                tool_id = tool_call_info.get("id")
                tool_name = tool_call_info.get("function", {}).get("name", "")
                raw_arguments = tool_call_info.get("function", {}).get("arguments", None)
                if isinstance(raw_arguments, str):
                    try:
                        args = json.loads(raw_arguments)
                    except:
                        args = {"raw_arguments": raw_arguments}
                else:
                    args = raw_arguments
                call_info = {"name": tool_name, "args": args, "id": tool_id, "type": "tool_call"}
                # Display only raw_arguments when available
                if raw_arguments is not None:
                    if isinstance(raw_arguments, str):
                        raw_display = raw_arguments
                    else:
                        raw_display = json.dumps(raw_arguments, indent=2, ensure_ascii=False)
                    entry = raw_display
                    accumulated_tool.append(entry)
                # Store tool call for later processing
                if tool_id and tool_name and tool_id not in seen_tool_call_ids:
                    seen_tool_call_ids.add(tool_id)
                    if "pending_tool_calls" not in st.session_state:
                        st.session_state.pending_tool_calls = []
                    st.session_state.pending_tool_calls.append({
                        "id": tool_id,
                        "name": tool_name,
                        "arguments": args
                    })
                    logging.debug(f"Added pending tool call from chunk: {tool_name} (id: {tool_id})")
                tool_placeholder.write("".join(accumulated_tool))
            
            # Simple string content
            elif isinstance(content, str):
                accumulated_text.append(content)
                text_placeholder.write("".join(accumulated_text))
                logging.debug(f"Added string content: {content[:100]}...")
            
            # Invalid tool calls
            elif (
                hasattr(message_content, "invalid_tool_calls")
                and message_content.invalid_tool_calls
            ):
                logging.debug("Found invalid_tool_calls in AIMessageChunk")
                tool_call_info = message_content.invalid_tool_calls[0]
                tool_str = json.dumps(tool_call_info, indent=2)
                entry = tool_str.replace("\n", "")
                accumulated_tool.append(entry)
                tool_placeholder.write("".join(accumulated_tool))
                logging.debug(f"Added invalid tool call info: {tool_str[:100]}...")
            
            # Tool call chunks
            elif (
                hasattr(message_content, "tool_call_chunks")
                and message_content.tool_call_chunks
            ):
                logging.debug("Found tool_call_chunks in AIMessageChunk")
                tool_call_chunk = message_content.tool_call_chunks[0]
                tool_str = str(tool_call_chunk)
                # entry = tool_str.replace("\n", "")
                accumulated_tool.append(entry)
                tool_placeholder.write("".join(accumulated_tool))
                logging.debug(f"Added tool call chunk: {tool_str[:100]}...")
            
            # Additional kwargs with tool calls
            elif (
                hasattr(message_content, "additional_kwargs")
                and "tool_calls" in message_content.additional_kwargs
                and message_content.additional_kwargs["tool_calls"]
            ):
                logging.debug("Found tool_calls in additional_kwargs")
                tool_calls = message_content.additional_kwargs["tool_calls"]
                for tool_call in tool_calls:
                    tool_id = tool_call.get("id")
                    tool_name = tool_call.get("function", {}).get("name", "")
                    raw_arguments = tool_call.get("function", {}).get("arguments", None)
                    if isinstance(raw_arguments, str):
                        try:
                            args = json.loads(raw_arguments)
                        except:
                            args = {"raw_arguments": raw_arguments}
                    else:
                        args = raw_arguments
                    call_info = {"name": tool_name, "args": args, "id": tool_id, "type": "tool_call"}
                    # Display only raw_arguments when available
                    if raw_arguments is not None:
                        if isinstance(raw_arguments, str):
                            raw_display = raw_arguments
                        else:
                            raw_display = json.dumps(raw_arguments, indent=2, ensure_ascii=False)
                        if tool_name:
                            entry = f"\n**도구 호출: {tool_name}**\n{raw_display}\n"
                        # else:
                        #     entry = raw_display.replace("\n", "")
                        accumulated_tool.append(entry)
                    if tool_id and tool_name and tool_id not in seen_tool_call_ids:
                        seen_tool_call_ids.add(tool_id)
                        if "pending_tool_calls" not in st.session_state:
                            st.session_state.pending_tool_calls = []
                        st.session_state.pending_tool_calls.append({
                            "id": tool_id,
                            "name": tool_name,
                            "arguments": args
                        })
                        logging.debug(f"Added pending tool call from kwargs: {tool_name} (id: {tool_id})")
                tool_placeholder.write("".join(accumulated_tool))
            
            else:
                logging.warning(f"Unhandled AIMessageChunk content format: {type(content)}")
        
        # Handle ToolMessage - this is the tool response
        elif isinstance(message_content, ToolMessage):
            logging.debug("Processing ToolMessage")
            tool_name = getattr(message_content, "name", "unknown_tool")
            tool_call_id = getattr(message_content, "tool_call_id", "unknown_id")
            content_str = str(message_content.content)
            
            logging.debug(f"ToolMessage received: name={tool_name}, tool_call_id={tool_call_id}")
            logging.debug(f"ToolMessage content: {content_str[:200]}...")
            
            # Store tool response for later processing
            if "tool_responses" not in st.session_state:
                st.session_state.tool_responses = {}
                
            st.session_state.tool_responses[tool_call_id] = {
                "name": tool_name,
                "content": content_str
            }
            
            # Find and remove from pending tools
            if "pending_tool_calls" in st.session_state:
                st.session_state.pending_tool_calls = [
                    call for call in st.session_state.pending_tool_calls 
                    if call.get("id") != tool_call_id
                ]
            
            # Add tool response JSON to the displayed tools
            try:
                response_data = json.loads(content_str)
                formatted = json.dumps(response_data, indent=2)
            except:
                formatted = content_str
            
            entry = f"\n\n**도구 호출 결과: {tool_name}**\n\n{formatted}\n"
            
            accumulated_tool.append(entry)
            tool_placeholder.write("".join(accumulated_tool))
            logging.debug(f"Added tool response content: {formatted[:100]}...")
        
        else:
            logging.warning(f"Unhandled message content type: {type(message_content)}")
        
        return None

    return callback_func, accumulated_text, accumulated_tool


# Handle tool execution
async def execute_tool(tool_call, tools):
    """Execute a tool and return its response"""
    tool_id = tool_call.get("id")
    tool_name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})
    
    logging.debug(f"Executing tool: {tool_name} (ID: {tool_id})")
    logging.debug(f"Arguments: {arguments}")
    
    # Find the matching tool
    matching_tool = None
    for tool in tools:
        if getattr(tool, "name", "") == tool_name:
            matching_tool = tool
            break
    
    if not matching_tool:
        error_msg = f"Tool {tool_name} not found"
        logging.error(error_msg)
        return {
            "tool_call_id": tool_id,
            "name": tool_name,
            "content": f"Error: {error_msg}"
        }
    
    try:
        # Execute the tool with provided arguments
        result = await matching_tool.ainvoke(arguments)
        logging.debug(f"Tool execution result: {str(result)[:200]}...")
        
        # Create response
        return {
            "tool_call_id": tool_id,
            "name": tool_name,
            "content": str(result)
        }
    except Exception as e:
        import traceback
        error_msg = f"Error executing tool {tool_name}: {str(e)}"
        error_trace = traceback.format_exc()
        logging.error(f"{error_msg}\n{error_trace}")
        return {
            "tool_call_id": tool_id,
            "name": tool_name,
            "content": f"Error: {error_msg}"
        }


# Log function entry and exit
logging.debug('Entering function: process_query')
async def process_query(query, text_placeholder, tool_placeholder, timeout_seconds=60):
    """
    사용자 질문을 처리하고 응답을 생성합니다.

    이 함수는 사용자의 질문을 에이전트에 전달하고, 응답을 실시간으로 스트리밍하여 표시합니다.
    지정된 시간 내에 응답이 완료되지 않으면 타임아웃 오류를 반환합니다.

    매개변수:
        query: 사용자가 입력한 질문 텍스트
        text_placeholder: 텍스트 응답을 표시할 Streamlit 컴포넌트
        tool_placeholder: 도구 호출 정보를 표시할 Streamlit 컴포넌트
        timeout_seconds: 응답 생성 제한 시간(초)

    반환값:
        response: 에이전트의 응답 객체
        final_text: 최종 텍스트 응답
        final_tool: 최종 도구 호출 정보
    """
    try:
        if st.session_state.agent:
            logging.debug(f"Processing query: {query}")
            streaming_callback, accumulated_text_obj, accumulated_tool_obj = (
                get_streaming_callback(text_placeholder, tool_placeholder)
            )
            
            # Reset tool tracking for new query
            st.session_state.pending_tool_calls = []
            st.session_state.tool_responses = {}
            
            try:
                # Log agent type and tools before processing
                logging.debug(f"Agent type: {type(st.session_state.agent)}")
                
                # Create runnable config with additional diagnostics
                config = RunnableConfig(
                    recursion_limit=st.session_state.recursion_limit,
                    thread_id=st.session_state.thread_id,
                    configurable={
                        "callbacks": [
                            # Add a simple callback for additional logging
                            lambda x: logging.debug(f"RunnableConfig callback: {str(x)[:100]}...")
                        ]
                    }
                )
                
                logging.debug(f"Starting agent execution with timeout: {timeout_seconds}s")
                
                # Start the agent execution
                agent_task = astream_graph(
                    st.session_state.agent,
                    {"messages": [HumanMessage(content=query)]},
                    callback=streaming_callback,
                    config=config,
                )
                
                # Process tool calls and agent responses
                has_tool_calls = False
                response = None
                
                # Use a timeout to prevent hanging
                try:
                    start_time = asyncio.get_event_loop().time()
                    remaining_time = timeout_seconds
                    
                    # Wait for initial response
                    response = await asyncio.wait_for(
                        agent_task,
                        timeout=remaining_time
                    )
                    
                    # Check for tool calls
                    logging.debug("Initial agent response received")
                    if "pending_tool_calls" in st.session_state and st.session_state.pending_tool_calls:
                        has_tool_calls = True
                        
                        # Process each pending tool call
                        while st.session_state.pending_tool_calls and remaining_time > 0:
                            logging.debug(f"Processing pending tool calls: {len(st.session_state.pending_tool_calls)}")
                            
                            # Get next tool call
                            tool_call = st.session_state.pending_tool_calls[0]
                            logging.debug(f"Processing tool call: {tool_call}")
                            
                            # Update remaining time
                            current_time = asyncio.get_event_loop().time()
                            elapsed = current_time - start_time
                            remaining_time = timeout_seconds - elapsed
                            
                            # Execute tool with timeout
                            if remaining_time <= 0:
                                logging.warning("Tool execution timeout")
                                break
                                
                            tool_result = await asyncio.wait_for(
                                execute_tool(tool_call, st.session_state.mcp_client.get_tools()),
                                timeout=remaining_time
                            )
                            
                            # Create tool message
                            logging.debug(f"Tool result: {str(tool_result)[:200]}...")
                            tool_message = ToolMessage(
                                content=tool_result["content"],
                                name=tool_result["name"],
                                tool_call_id=tool_result["tool_call_id"]
                            )
                            
                            # Display tool result
                            with tool_placeholder.expander("🔧 도구 실행 결과", expanded=True):
                                st.write(f"**도구**: {tool_result['name']}\n\n**결과**:\n```\n{tool_result['content'][:1000]}...\n```")
                            
                            # Update remaining time
                            current_time = asyncio.get_event_loop().time()
                            elapsed = current_time - start_time
                            remaining_time = timeout_seconds - elapsed
                            
                            # Pass tool result back to agent
                            if remaining_time <= 0:
                                logging.warning("Agent continuation timeout")
                                break
                                
                            agent_continue_task = astream_graph(
                                st.session_state.agent,
                                {"messages": [tool_message]},
                                callback=streaming_callback,
                                config=config,
                            )
                            
                            # Wait for agent response
                            response = await asyncio.wait_for(
                                agent_continue_task,
                                timeout=remaining_time
                            )
                            
                            # Remove processed tool call
                            st.session_state.pending_tool_calls = st.session_state.pending_tool_calls[1:]
                            
                            # Update remaining time
                            current_time = asyncio.get_event_loop().time()
                            elapsed = current_time - start_time
                            remaining_time = timeout_seconds - elapsed
                            
                            # Break if no more pending calls
                            if not st.session_state.pending_tool_calls:
                                logging.debug("No more pending tool calls")
                                break
                    
                except asyncio.TimeoutError:
                    error_msg = f"⏱️ 요청 시간이 {timeout_seconds}초를 초과했습니다. 나중에 다시 시도해 주세요."
                    logging.error(f"Query timed out after {timeout_seconds} seconds")
                    return {"error": error_msg}, error_msg, ""
                
                logging.debug("Query completed successfully")
                
                # Log response details
                if hasattr(response, 'get'):
                    resp_content = response.get('content', 'No content')
                    logging.debug(f"Response content: {str(resp_content)[:100]}...")
                else:
                    logging.debug(f"Response type: {type(response)}")
                
            except asyncio.TimeoutError:
                error_msg = f"⏱️ 요청 시간이 {timeout_seconds}초를 초과했습니다. 나중에 다시 시도해 주세요."
                logging.error(f"Query timed out after {timeout_seconds} seconds")
                return {"error": error_msg}, error_msg, ""
            except Exception as e:
                import traceback
                error_msg = f"쿼리 처리 중 오류 발생: {str(e)}"
                error_trace = traceback.format_exc()
                logging.error(f"{error_msg}\n{error_trace}")
                return {"error": error_msg}, error_msg, error_trace

            # Process final results
            final_text = "".join(accumulated_text_obj)
            final_tool = "".join(accumulated_tool_obj)
            
            # Log final output for debugging
            logging.debug(f"Final text length: {len(final_text)}")
            logging.debug(f"Final text: {final_text[:100]}...")
            logging.debug(f"Final tool content length: {len(final_tool)}")
            logging.debug(f"Final tool content: {final_tool[:100]}...")
            
            return response, final_text, final_tool
        else:
            logging.warning("Agent not initialized before query")
            return (
                {"error": "🚫 에이전트가 초기화되지 않았습니다."},
                "🚫 에이전트가 초기화되지 않았습니다.",
                "",
            )
    except Exception as e:
        import traceback
        error_msg = f"❌ 쿼리 처리 중 오류 발생: {str(e)}"
        error_trace = traceback.format_exc()
        logging.error(f"{error_msg}\n{error_trace}")
        return {"error": error_msg}, error_msg, error_trace





def load_selected_prompt():
    selected = st.session_state["prompt_selectbox"]
    prompts_dict = prompt_data.get("prompts", {})
    if selected in prompts_dict:
        st.session_state.selected_prompt_name = selected
        st.session_state.selected_prompt_text = prompts_dict[selected]["prompt"]
        st.session_state.prompt_loaded = True
        st.session_state["sidebar_edit_prompt_text"] = prompts_dict[selected]["prompt"]


def load_selected_tool():
    selected = st.session_state["tool_selectbox"]
    logging.debug(f"Selected tool: {selected}")
    selected_tool = next((t for t in tools_list if t["name"] == selected), None)
    if selected_tool:
        logging.debug(f"Loading tool configuration from: {selected_tool['path']}")
        try:
            with open(selected_tool["path"], encoding="utf-8") as f:
                st.session_state.tool_config = json.load(f)
            st.session_state.file_path = selected_tool["path"]
            st.session_state.loaded = True
            # Normalize pending MCP config: only keep valid connection fields
            raw_conf = st.session_state.tool_config.get("mcpServers", st.session_state.tool_config)
            pending_conf = {}
            for srv_name, srv_cfg in raw_conf.items():
                if "url" in srv_cfg:
                    # SSE connection
                    conf = {"transport": srv_cfg.get("transport", "sse"), "url": srv_cfg["url"]}
                    if "headers" in srv_cfg:
                        conf["headers"] = srv_cfg["headers"]
                    if "timeout" in srv_cfg:
                        conf["timeout"] = srv_cfg["timeout"]
                    if "sse_read_timeout" in srv_cfg:
                        conf["sse_read_timeout"] = srv_cfg["sse_read_timeout"]
                    if "session_kwargs" in srv_cfg:
                        conf["session_kwargs"] = srv_cfg["session_kwargs"]
                else:
                    # stdio connection
                    conf = {"transport": srv_cfg.get("transport", "stdio"), "command": srv_cfg["command"], "args": srv_cfg["args"]}
                    if "env" in srv_cfg:
                        conf["env"] = srv_cfg["env"]
                    if "cwd" in srv_cfg:
                        conf["cwd"] = srv_cfg["cwd"]
                    if "encoding" in srv_cfg:
                        conf["encoding"] = srv_cfg["encoding"]
                    if "encoding_error_handler" in srv_cfg:
                        conf["encoding_error_handler"] = srv_cfg["encoding_error_handler"]
                    if "session_kwargs" in srv_cfg:
                        conf["session_kwargs"] = srv_cfg["session_kwargs"]
                pending_conf[srv_name] = conf
            # Store direct mapping for initialization (initialize_session will unpack it)
            st.session_state.pending_mcp_config = pending_conf
            logging.debug("Tool configuration loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading tool configuration: {str(e)}")




# 세션 상태 초기화
if "session_initialized" not in st.session_state:
    logging.debug('Session state not initialized, setting default values')
    st.session_state.session_initialized = False  # 세션 초기화 상태 플래그
    st.session_state.agent = None  # ReAct 에이전트 객체 저장 공간
    st.session_state.history = []  # 대화 기록 저장 리스트
    st.session_state.mcp_client = None  # MCP 클라이언트 객체 저장 공간
    st.session_state.timeout_seconds = 180  # 응답 생성 제한 시간(초), 기본값 120초
    st.session_state.selected_model = "gpt-4o"  # 기본 모델 선택
    st.session_state.recursion_limit = 100  # 재귀 호출 제한, 기본값 100
    st.session_state.selected_prompt_text = ""  # initialize selected prompt text
    st.session_state.temperature = 0.1  # 기본 temperature 설정
    st.session_state.pending_tool_calls = []  # 대기 중인 도구 호출 목록
    st.session_state.tool_responses = {}  # 도구 응답 저장 공간
    # Load default system prompt if none selected
    try:
        with open("prompts/system_prompt.yaml", "r", encoding="utf-8") as f:
            sys_data = yaml.safe_load(f)
            default_prompt = sys_data.get("template", "")
            # store system prompt separately for tool usage and initialize selected prompt
            st.session_state.system_prompt_text = default_prompt
            st.session_state.selected_prompt_text = default_prompt
    except Exception as e:
        logging.warning(f"Failed to load system prompt: {e}")

    # Auto-load AI App settings from URL 'id' param
    query_params = st.query_params
    if "id" in query_params:
        app_id = query_params["id"]
        if not st.session_state.get("auto_loaded", False):
            try:
                with open("store/ai_app_store.json", "r", encoding="utf-8") as f:
                    ai_app_store = json.load(f)
                for section in ai_app_store.get("AIAppStore", []):
                    for app in section.get("apps", []):
                        url_parts = urlsplit(app.get("url", ""))
                        params = parse_qs(url_parts.query)
                        if params.get("id", [None])[0] == app_id:
                            st.session_state.selected_model = app.get("model", st.session_state.selected_model)
                            st.session_state.temperature = app.get("temperature", st.session_state.temperature)
                            prompt_text = app.get("prompt", "")
                            if prompt_text:
                                st.session_state.selected_prompt_text = prompt_text
                                st.session_state.sidebar_edit_prompt_text = prompt_text
                            tool_config = app.get("tools", {})
                            st.session_state.tool_config = tool_config
                            raw_conf = tool_config.get("mcpServers", tool_config)
                            pending_conf = {}
                            for srv_name, srv_cfg in raw_conf.items():
                                if "url" in srv_cfg:
                                    conf = {"transport": srv_cfg.get("transport", "sse"), "url": srv_cfg["url"]}
                                    for k in ["headers", "timeout", "sse_read_timeout", "session_kwargs"]:
                                        if k in srv_cfg:
                                            conf[k] = srv_cfg[k]
                                else:
                                    conf = {"transport": srv_cfg.get("transport", "stdio"), "command": srv_cfg.get("command"), "args": srv_cfg.get("args")}
                                    for k in ["env", "cwd", "encoding", "encoding_error_handler", "session_kwargs"]:
                                        if k in srv_cfg:
                                            conf[k] = srv_cfg[k]
                                pending_conf[srv_name] = conf
                            st.session_state.pending_mcp_config = pending_conf
                            st.session_state.auto_loaded = True
                            st.session_state.prompt_loaded = True
                            st.session_state.prompt_selectbox = ""
                            st.session_state.tool_selectbox = ""
                            st.session_state.loaded = True
                            st.session_state.app_title = app.get("title", "Universal Agent")
                            success = st.session_state.event_loop.run_until_complete(initialize_session(st.session_state.pending_mcp_config))
                            st.session_state.session_initialized = success
                            if success:
                                st.rerun()
                            break
                        if st.session_state.get("auto_loaded", False):
                            break
            except Exception as e:
                st.error(f"Error loading AI App config: {e}")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = random_uuid()

try:
    # Suppress async generator cleanup errors
    sys.set_asyncgen_hooks(finalizer=lambda agen: None)
except AttributeError as e:
    logging.error(f'AttributeError: {str(e)}')




# Load MCP config JSON paths for tools selection
MCP_CONFIG_DIR = "mcp-config"
os.makedirs(MCP_CONFIG_DIR, exist_ok=True)
json_paths = glob.glob(f"{MCP_CONFIG_DIR}/*.json")
if not json_paths and not os.path.exists(f"{MCP_CONFIG_DIR}/mcp_config.json"):
    default_config = {"mcpServers": {}}
    with open(f"{MCP_CONFIG_DIR}/mcp_config.json", "w", encoding="utf-8") as f:
        json.dump(default_config, f, indent=2, ensure_ascii=False)
    json_paths = [f"{MCP_CONFIG_DIR}/mcp_config.json"]

st.sidebar.markdown("##### 💡 Store에서 장바구니에 담은 Prompt와 MCP Tool을 조합하여 나만의 AI Agent를 만들어 보세요.")

# --- Prompt Store (프롬프트 선택 및 관리) ---
# EMP_NO 기반 프롬프트 경로 설정
PROMPT_CONFIG_DIR = "prompt-config"
logging.debug('Loading configuration from .env')
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
EMP_NO = os.getenv("EMP_NO", "default_emp_no")
EMP_NAME = os.getenv("EMP_NAME", "default_emp_name")
PROMPT_STORE_PATH = os.path.join(PROMPT_CONFIG_DIR, f"{EMP_NO}.json")

# 프롬프트 파일이 없으면 안내 메시지 출력
if not os.path.exists(PROMPT_STORE_PATH):
    st.sidebar.warning(f"{PROMPT_STORE_PATH} 파일이 없습니다. Prompt Store에서 장바구니 저장을 먼저 해주세요.")
    prompt_data = {"prompts": {}}
else:
    with open(PROMPT_STORE_PATH, encoding="utf-8") as f:
        prompt_data = json.load(f)


# --- Sidebar for File Selection, Save, and Tool List ---
with st.sidebar:
    
    st.selectbox(
        "모델 선택",
        options=list(OUTPUT_TOKEN_INFO.keys()),
        key="selected_model",
    )
    st.slider(
        "Temperature",
        min_value=0.0,
        max_value=0.5,
        step=0.01,
        key="temperature",
        help="Temperature: 낮을수록 고정된 답변, 높을수록 창의적인 답변",
    )

    prompts_dict = prompt_data.get("prompts", {})
    prompt_names = list(prompts_dict.keys()) if prompts_dict else ["(등록된 프롬프트 없음)"]
    if st.session_state.get("auto_loaded", False):
        prompt_names = [""] + prompt_names
    st.selectbox(
        "프롬프트 선택",
        prompt_names,
        key="prompt_selectbox",
        on_change=load_selected_prompt,
    )
    if st.session_state.get("auto_loaded", False):
        new_prompt_text = st.text_area("프롬프트 내용", key="sidebar_edit_prompt_text", height=120)
    else:
        # Ensure selected prompt loaded on initial render
        if "prompt_loaded" not in st.session_state or not st.session_state.prompt_loaded:
            load_selected_prompt()

    # 프롬프트 선택 시 바로 아래에 내용 보여주고 수정/저장 가능하게
    selected_prompt = st.session_state.get("prompt_selectbox")
    if selected_prompt and selected_prompt in prompts_dict:
        # Ensure initial sidebar prompt text is set, then bind text_area to session state without default value
        if "sidebar_edit_prompt_text" not in st.session_state:
            st.session_state.sidebar_edit_prompt_text = prompts_dict[selected_prompt]["prompt"]
        new_prompt_text = st.text_area("프롬프트 내용", key="sidebar_edit_prompt_text", height=120)
        if "share_mode" not in st.session_state:
            st.session_state.share_mode = False
        if "clear_share_prompt_title" not in st.session_state:
            st.session_state.clear_share_prompt_title = False
        if st.button("📤 프롬프트 공유", key="sidebar_share_prompt", use_container_width=True):
            st.session_state.share_mode = True
            if st.session_state.share_mode:
                new_title = st.text_input(
                    "공유할 프롬프트 제목을 입력하세요",
                    key="share_prompt_title",
                    value="" if st.session_state.clear_share_prompt_title else st.session_state.get("share_prompt_title", "")
                )
                if st.button("공유", key="share_prompt_confirm", use_container_width=True):
                    global_prompt_store_path = os.path.join("store", "prompt_store.json")
                    if os.path.exists(global_prompt_store_path):
                        with open(global_prompt_store_path, encoding="utf-8") as f:
                            global_prompt_data = json.load(f)
                    else:
                        global_prompt_data = {"prompts": {}}
                    global_prompts_dict = global_prompt_data.get("prompts", {})
                    if not new_title.strip():
                        st.warning("제목을 입력해주세요.")
                    elif new_title in global_prompts_dict:
                        st.warning(f"이미 존재하는 제목입니다: {new_title}. 다른 제목을 입력해주세요.")
                    else:
                        global_prompt_data["prompts"][new_title] = {"prompt": new_prompt_text, "EMP_NO": EMP_NO, "EMP_NAME": EMP_NAME}
                        with open(global_prompt_store_path, "w", encoding="utf-8") as f:
                            json.dump(global_prompt_data, f, indent=2, ensure_ascii=False)
                        st.session_state.saved_msg = f"{new_title} 프롬프트 공유 완료"
                        st.session_state.share_mode = False
                        st.session_state.clear_share_prompt_title = True
                        st.rerun()
                else:
                    st.session_state.clear_share_prompt_title = False
            else:
                st.session_state.clear_share_prompt_title = False


    tools_list = [{"name": Path(p).stem, "path": p} for p in json_paths]
    tool_names = [t["name"] for t in tools_list]
    default_tool_index = 0
    if "file_path" in st.session_state:
        current_name = Path(st.session_state.file_path).stem
        if current_name in tool_names:
            default_tool_index = tool_names.index(current_name)
    if st.session_state.get("auto_loaded", False):
        tool_names = [""] + tool_names
        default_tool_index = 0
    st.selectbox(
        "MCP Tool 목록 선택",
        tool_names,
        key="tool_selectbox",
        index=default_tool_index,
        on_change=load_selected_tool,
    )
    # Load default tool configuration on initial render if not already loaded
    if not st.session_state.get("auto_loaded", False) and not st.session_state.get("loaded", False):
        load_selected_tool()

    # Tool 목록 (List & Delete)
    if st.session_state.get("loaded", False):
        mcp = st.session_state.tool_config.get("mcpServers", {})

        st.markdown("MCP Tool 목록")
        if not mcp:
            st.warning("등록된 도구가 없습니다.")
        else:
            for name in list(mcp.keys()):
                st.write(f"• {name}")


    # 에이전트 설정 적용 및 대화 초기화 버튼 추가
    if st.button("Agent 생성하기", key="create_agent_button", type="primary", use_container_width=True):
        apply_status = st.empty()
        with apply_status.container():
            st.warning("🔄 에이전트를 생성 중입니다... 잠시만 기다려주세요.")
            progress_bar = st.progress(0)
            # 세션 초기화
            st.session_state.session_initialized = False
            st.session_state.agent = None
            progress_bar.progress(30)
            # 초기화 실행
            success = st.session_state.event_loop.run_until_complete(
                initialize_session(st.session_state.pending_mcp_config)
            )
            progress_bar.progress(100)
            if success:
                st.success("✅ 에이전트가 생성되었습니다.")
            else:
                st.error("❌ 에이전트 생성에 실패하였습니다.")
            # 에이전트 초기화 완료 상태를 강제로 설정하여 채팅을 활성화합니다
            st.session_state.session_initialized = True
        # 페이지 새로고침
        st.rerun()
    if st.button("💬 대화 초기화", key="reset_chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.thread_id = random_uuid()
        st.rerun()
    # Add AI App Store registration UI
    if st.button("AI App Store 등록", key="aiapp_register_sidebar", use_container_width=True):
        st.session_state.show_aiapp_registration = True
        # Reset AI App form fields when opening the form
        st.session_state.aiapp_new_name = ""
        st.session_state.aiapp_new_desc = ""
        st.session_state.aiapp_new_url = ""
        st.session_state.aiapp_new_hash_tags = ""
        # Remove any previous upload to avoid session_state conflict
        st.session_state.pop("aiapp_icon_upload", None)

    if st.session_state.get("show_aiapp_registration", False):
        st.subheader("신규 AI App 등록")
        aiapp_name = st.text_input("App 이름", value=st.session_state.get("aiapp_new_name", ""), key="aiapp_new_name")
        aiapp_desc = st.text_input("App 설명", value=st.session_state.get("aiapp_new_desc", ""), key="aiapp_new_desc")
        # Optional icon upload
        aiapp_icon_file = st.file_uploader("App 아이콘 (선택, PNG/JPG)", type=["png","jpg","jpeg"], key="aiapp_icon_upload")
        # URL input with validation
        aiapp_url = st.text_input("App ID (예: Search_Agent)", value=st.session_state.get("aiapp_new_url", ""), key="aiapp_new_url")
        # Optional hashtags input
        aiapp_hash_tags = st.text_input("Hash Tags (콤마로 구분)", value=st.session_state.get("aiapp_new_hash_tags", ""), key="aiapp_new_hash_tags")
        if st.button("등록", key="aiapp_submit_btn", use_container_width=True):
            AI_APP_STORE_PATH = "store/ai_app_store.json"
            # Load existing AI App store
            if not os.path.exists(AI_APP_STORE_PATH) or os.path.getsize(AI_APP_STORE_PATH) == 0:
                apps_by_type = {"auto": [], "user": []}
            else:
                with open(AI_APP_STORE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                apps_by_type = {"auto": [], "user": []}
                for section in data.get("AIAppStore", []):
                    t = section.get("type")
                    if t in apps_by_type:
                        apps_by_type[t].extend(section.get("apps", []))
            # Gather inputs
            name = st.session_state.aiapp_new_name.strip()
            desc = st.session_state.aiapp_new_desc.strip()
            url = f"{URL_BASE}{st.session_state.aiapp_new_url.strip()}"
            icon_file = aiapp_icon_file
            tags_raw = st.session_state.aiapp_new_hash_tags.strip()
            hash_tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
            # Check duplicates
            existing_names = [app.get("title", "") for app in apps_by_type["auto"] + apps_by_type["user"]]
            existing_urls = [app.get("url", "") for app in apps_by_type["auto"] + apps_by_type["user"] if app.get("url")]
            # Validate mandatory fields
            if not name or not desc or not url:
                st.error("이름, 설명, URL을 모두 입력하세요.")
            elif name in existing_names:
                st.error(f"이미 존재하는 App 이름입니다. (중복: {name})")
            else:
                # Validate URL format and characters
                parsed = urlsplit(url)
                if not parsed.scheme or not parsed.netloc:
                    st.error("유효한 URL을 입력하세요. 예: http://example.com")
                elif any(c in url for c in ['"', "'", '<', '>', ' ']):
                    st.error("URL에 허용되지 않는 특수문자가 포함되어 있습니다.")
                elif url in existing_urls:
                    st.error(f"이미 등록된 URL입니다: {url}")
                else:
                    # Save icon if provided
                    icon_name = None
                    if icon_file:
                        ext = Path(icon_file.name).suffix
                        icon_name = f"aiapp_{name}{ext}"
                        icon_dir = os.path.join(ASSETS_DIR, "icons")
                        os.makedirs(icon_dir, exist_ok=True)
                        save_path = os.path.join(icon_dir, icon_name)
                        with open(save_path, "wb") as f:
                            f.write(icon_file.read())
                    # Append new entry
                    apps_by_type["auto"].append({
                        "prompt": st.session_state.selected_prompt_text,
                        "tools": st.session_state.tool_config,
                        "model": st.session_state.selected_model,
                        "temperature": st.session_state.temperature,
                        "title": name,
                        "icon_name": icon_name,
                        "url": url,
                        "hash_tag": hash_tags,
                        "like": 0,
                        "EMP_NO": EMP_NO,
                        "EMP_NAME": EMP_NAME,
                        "description": desc
                    })
                    # Save updated store
                    new_data = {"AIAppStore": [
                        {"type": "auto", "apps": apps_by_type["auto"]},
                        {"type": "user", "apps": apps_by_type["user"]}
                    ]}
                    with open(AI_APP_STORE_PATH, "w", encoding="utf-8") as f:
                        json.dump(new_data, f, indent=2, ensure_ascii=False)
                    st.success("신규 AI App이 등록되었습니다! 새로고침 후 확인하세요.")
                    st.session_state.show_aiapp_registration = False


# --- Main Area ---
title_text = f"🤖 {st.session_state.get('app_title', 'Universal Agent')}"
st.title(title_text)
st.markdown("---")

# 하단 저장 메시지 출력
with st.sidebar:
    if st.session_state.get("saved_msg"):
        st.success(st.session_state.pop("saved_msg"))

# Monkey-patch MultiServerMCPClient.__aexit__ to suppress 'no running event loop' errors during cleanup
_orig_mcp_aexit = MultiServerMCPClient.__aexit__
async def _safe_mcp_aexit(self, exc_type, exc_val, exc_tb):
    try:
        await _orig_mcp_aexit(self, exc_type, exc_val, exc_tb)
    except RuntimeError:
        # Suppress errors when event loop is closed
        pass
    except Exception:
        # Suppress any cleanup-related errors
        pass
MultiServerMCPClient.__aexit__ = _safe_mcp_aexit

# Monkey-patch stdio_client to suppress 'no running event loop' errors
_orig_stdio_client = _stdio.stdio_client
@asynccontextmanager
async def safe_stdio_client(server, errlog=sys.stderr):
    try:
        async with _orig_stdio_client(server, errlog) as (read_stream, write_stream):
            yield read_stream, write_stream
    except RuntimeError:
        # Suppress errors when event loop is closed
        pass
    except Exception:
        # Suppress any cleanup-related errors
        pass
# Override stdio_client with the safe version
_stdio.stdio_client = safe_stdio_client

# Also patch the imported stdio_client in langchain_mcp_adapters.client
import langchain_mcp_adapters.client as _lcmcp_client
_lcmcp_client.stdio_client = safe_stdio_client

# Auto-load AI App settings from URL 'id' param
query_params = st.query_params
if "id" in query_params:
    app_id = query_params["id"]
    if not st.session_state.get("auto_loaded", False):
        try:
            with open("store/ai_app_store.json", "r", encoding="utf-8") as f:
                ai_app_store = json.load(f)
            for section in ai_app_store.get("AIAppStore", []):
                for app in section.get("apps", []):
                    url_parts = urlsplit(app.get("url", ""))
                    params = parse_qs(url_parts.query)
                    if params.get("id", [None])[0] == app_id:
                        # apply app config
                        st.session_state.selected_model = app.get("model", st.session_state.selected_model)
                        st.session_state.temperature = app.get("temperature", st.session_state.temperature)
                        prompt_text = app.get("prompt", "")
                        if prompt_text:
                            st.session_state.selected_prompt_text = prompt_text
                            st.session_state.sidebar_edit_prompt_text = prompt_text
                        tool_config = app.get("tools", {})
                        st.session_state.tool_config = tool_config
                        raw_conf = tool_config.get("mcpServers", tool_config)
                        pending_conf = {}
                        for srv_name, srv_cfg in raw_conf.items():
                            if "url" in srv_cfg:
                                conf = {"transport": srv_cfg.get("transport", "sse"), "url": srv_cfg["url"]}
                                for k in ["headers", "timeout", "sse_read_timeout", "session_kwargs"]:
                                    if k in srv_cfg:
                                        conf[k] = srv_cfg[k]
                            else:
                                conf = {"transport": srv_cfg.get("transport", "stdio"), "command": srv_cfg.get("command"), "args": srv_cfg.get("args")}
                                for k in ["env", "cwd", "encoding", "encoding_error_handler", "session_kwargs"]:
                                    if k in srv_cfg:
                                        conf[k] = srv_cfg[k]
                            pending_conf[srv_name] = conf
                        st.session_state.pending_mcp_config = pending_conf
                        st.session_state.auto_loaded = True
                        st.session_state.prompt_loaded = True
                        st.session_state.prompt_selectbox = ""
                        st.session_state.tool_selectbox = ""
                        st.session_state.loaded = True
                        st.session_state.app_title = app.get("title", "Universal Agent")
                        success = st.session_state.event_loop.run_until_complete(initialize_session(st.session_state.pending_mcp_config))
                        st.session_state.session_initialized = success
                        if success:
                            st.rerun()
                        break
                if st.session_state.get("auto_loaded", False):
                    break
        except Exception as e:
            st.error(f"Error loading AI App config: {e}")

# --- Main Chat Area ---
if not st.session_state.session_initialized:
    st.info("⚠️ MCP 서버와 Agent가 초기화되지 않았습니다. 왼쪽 사이드바에서 Prompt와 MCP Tool을 선택하고 'Agent 생성하기' 버튼을 클릭하여 초기화해주세요.")

# --- 대화 기록 출력 ---
print_message()

# --- 사용자 입력 및 처리 ---
user_query = st.session_state.pop("user_query", None) or st.chat_input("💬 질문을 입력하세요")
if user_query:
    if st.session_state.session_initialized:
        st.chat_message("user", avatar="🧑🏻").write(user_query)
        with st.chat_message("assistant", avatar="🤖"):
            tool_placeholder = st.empty()
            text_placeholder = st.empty()
            
            # 진행 상태 표시기 추가
            progress_placeholder = st.empty()
            with progress_placeholder.container():
                st.write("🔍 Agent가 도구를 통해 답변을 찾고 있습니다...")
                progress_bar = st.progress(0)
                
                # 도구 호출 및 처리 상태 모니터링
                def update_progress():
                    pending_count = len(st.session_state.get("pending_tool_calls", []))
                    if pending_count > 0:
                        return 30  # 도구 호출 시작
                    
                    response_count = len(st.session_state.get("tool_responses", {}))
                    if response_count > 0:
                        return 70  # 도구 응답 처리 중
                        
                    return 100  # 완료
                
                # 즉시 진행 상태 업데이트
                progress_bar.progress(10)
            
            # 실제 쿼리 처리
            resp, final_text, final_tool = (
                st.session_state.event_loop.run_until_complete(
                    process_query(
                        user_query,
                        text_placeholder,
                        tool_placeholder,
                        st.session_state.timeout_seconds,
                    )
                )
            )
            
            # 진행 표시기 제거
            progress_placeholder.empty()
            
        if "error" in resp:
            st.error(resp["error"])
        else:
            st.session_state.history.append({"role": "user", "content": user_query})
            # Generate follow-up questions and add to assistant message
            openai_api_key = os.getenv("OPENAI_API_KEY", "")
            raw_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com")
            parsed = urlsplit(raw_base)
            openai_api_base = f"{parsed.scheme}://{parsed.netloc}/v1"
            followup_llm = get_followup_llm(
                st.session_state.selected_model,
                0.3,
                openai_api_key,
                openai_api_base,
            )
            followups = st.session_state.event_loop.run_until_complete(
                generate_followups(followup_llm, final_text)
            )
            st.session_state.history.append(
                {"role": "assistant", "content": final_text, "followups": followups}
            )
            if final_tool.strip():
                st.session_state.history.append(
                    {"role": "assistant_tool", "content": final_tool}
                )
            st.rerun()
    else:
        st.warning(
            "⚠️ MCP 서버와 Agent가 초기화되지 않았습니다. 왼쪽 사이드바에서 Prompt와 MCP Tool을 선택하고 'Agent 생성하기' 버튼을 클릭하여 초기화해주세요."
        )

