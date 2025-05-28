from __future__ import annotations
from typing import Any, Dict, List, Callable, Optional, Sequence
from dataclasses import dataclass
import uuid

from langchain_core.messages import (
    AIMessageChunk,
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage
)
from langchain_core.runnables import RunnableConfig
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_openai import ChatOpenAI

# --- Added helper to normalize streamed chunk content ---
def _extract_chunk_content(chunk_msg):
    """Return a printable representation for streamed chunks.

    * For LangChain BaseMessage types, prefer ``.content``.  
    * If content is empty, fall back to ``additional_kwargs['tool_output']`` or
      serialized ``additional_kwargs['function_call']`` so that tool responses
      are rendered in the UI instead of an empty string.
    """
    try:
        from langchain_core.messages import BaseMessage  # type: ignore
    except Exception:
        # Fallback import for older langchain
        try:
            from langchain.schema import BaseMessage  # type: ignore
        except Exception:
            BaseMessage = None  # type: ignore

    if BaseMessage and isinstance(chunk_msg, BaseMessage):
        # Prefer the normal textual content
        if getattr(chunk_msg, "content", None):
            return chunk_msg.content
        # Fallbacks for function/tool call outputs
        ak = getattr(chunk_msg, "additional_kwargs", {}) or {}
        if isinstance(ak, dict):
            if "tool_output" in ak:
                return ak["tool_output"]
            if "function_call" in ak:
                import json as _json
                try:
                    return _json.dumps(ak["function_call"], ensure_ascii=False)
                except Exception:
                    pass
        # Last resort: str()
        return str(chunk_msg)
    # Non‑message types: str‑cast
    return str(chunk_msg)

from langgraph.graph.state import CompiledStateGraph


def random_uuid():
    return str(uuid.uuid4())


async def astream_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: Optional[RunnableConfig] = None,
    node_names: List[str] = [],
    callback: Optional[Callable] = None,
    stream_mode: str = "messages",
    include_subgraphs: bool = False,
) -> Dict[str, Any]:
    """
    LangGraph의 실행 결과를 비동기적으로 스트리밍하고 직접 출력하는 함수입니다.

    Args:
        graph (CompiledStateGraph): 실행할 컴파일된 LangGraph 객체
        inputs (dict): 그래프에 전달할 입력값 딕셔너리
        config (Optional[RunnableConfig]): 실행 설정 (선택적)
        node_names (List[str], optional): 출력할 노드 이름 목록. 기본값은 빈 리스트
        callback (Optional[Callable], optional): 각 청크 처리를 위한 콜백 함수. 기본값은 None
            콜백 함수는 {"node": str, "content": Any} 형태의 딕셔너리를 인자로 받습니다.
        stream_mode (str, optional): 스트리밍 모드 ("messages" 또는 "updates"). 기본값은 "messages"
        include_subgraphs (bool, optional): 서브그래프 포함 여부. 기본값은 False

    Returns:
        Dict[str, Any]: 최종 결과 (선택적)
    """
    config = config or {}
    final_result = {}

    def format_namespace(namespace):
        return namespace[-1].split(":")[0] if len(namespace) > 0 else "root graph"

    prev_node = ""

    if stream_mode == "messages":
        async for chunk_msg, metadata in graph.astream(
            inputs, config, stream_mode=stream_mode
        ):
            curr_node = metadata["langgraph_node"]
            final_result = {"node": curr_node, "content": _extract_chunk_content(chunk_msg), "metadata": metadata}

            # node_names가 비어있거나 현재 노드가 node_names에 있는 경우에만 처리
            if not node_names or curr_node in node_names:
                # 콜백 함수가 있는 경우 실행
                if callback:
                    result = callback({"node": curr_node, "content": _extract_chunk_content(chunk_msg)})
                    if hasattr(result, "__await__"):
                        await result
                # 콜백이 없는 경우 기본 출력
                else:
                    # 노드가 변경된 경우에만 구분선 출력
                    if curr_node != prev_node:
                        print("\n" + "=" * 50)
                        print(f"🔄 Node: \033[1;36m{curr_node}\033[0m 🔄")
                        print("- " * 25)
                    
                    # Claude/Anthropic 모델의 토큰 청크 처리 - 항상 텍스트만 추출
                    if hasattr(chunk_msg, 'content'):
                        # 리스트 형태의 content (Anthropic/Claude 스타일)
                        if isinstance(chunk_msg.content, list):
                            for item in chunk_msg.content:
                                if isinstance(item, dict) and 'text' in item:
                                    print(item['text'], end="", flush=True)
                        # 문자열 형태의 content
                        elif isinstance(chunk_msg.content, str):
                            print(chunk_msg.content, end="", flush=True)
                    # 그 외 형태의 chunk_msg 처리
                    else:
                        print(chunk_msg, end="", flush=True)

                prev_node = curr_node

    elif stream_mode == "updates":
        # 에러 수정: 언패킹 방식 변경
        # REACT 에이전트 등 일부 그래프에서는 단일 딕셔너리만 반환함
        async for chunk in graph.astream(
            inputs, config, stream_mode=stream_mode, subgraphs=include_subgraphs
        ):
            # 반환 형식에 따라 처리 방법 분기
            if isinstance(chunk, tuple) and len(chunk) == 2:
                # 기존 예상 형식: (namespace, chunk_dict)
                namespace, node_chunks = chunk
            else:
                # 단일 딕셔너리만 반환하는 경우 (REACT 에이전트 등)
                namespace = []  # 빈 네임스페이스 (루트 그래프)
                node_chunks = chunk  # chunk 자체가 노드 청크 딕셔너리
            
            # 딕셔너리인지 확인하고 항목 처리
            if isinstance(node_chunks, dict):
                for node_name, node_chunk in node_chunks.items():
                    final_result = {"node": node_name, "content": node_chunk, "namespace": namespace}
                    
                    # node_names가 비어있지 않은 경우에만 필터링
                    if len(node_names) > 0 and node_name not in node_names:
                        continue

                    # 콜백 함수가 있는 경우 실행
                    if callback is not None:
                        result = callback({"node": node_name, "content": node_chunk})
                        if hasattr(result, "__await__"):
                            await result
                    # 콜백이 없는 경우 기본 출력
                    else:
                        # 노드가 변경된 경우에만 구분선 출력 (messages 모드와 동일하게)
                        if node_name != prev_node:
                            print("\n" + "=" * 50)
                            print(f"🔄 Node: \033[1;36m{node_name}\033[0m 🔄")
                            print("- " * 25)
                        
                        # 노드의 청크 데이터 출력 - 텍스트 중심으로 처리
                        if isinstance(node_chunk, dict):
                            for k, v in node_chunk.items():
                                if isinstance(v, BaseMessage):
                                    # BaseMessage의 content 속성이 텍스트나 리스트인 경우를 처리
                                    if hasattr(v, 'content'):
                                        if isinstance(v.content, list):
                                            for item in v.content:
                                                if isinstance(item, dict) and 'text' in item:
                                                    print(item['text'], end="", flush=True)
                                        else:
                                            print(v.content, end="", flush=True)
                                    else:
                                        v.pretty_print()
                                elif isinstance(v, list):
                                    for list_item in v:
                                        if isinstance(list_item, BaseMessage):
                                            if hasattr(list_item, 'content'):
                                                if isinstance(list_item.content, list):
                                                    for item in list_item.content:
                                                        if isinstance(item, dict) and 'text' in item:
                                                            print(item['text'], end="", flush=True)
                                                else:
                                                    print(list_item.content, end="", flush=True)
                                            else:
                                                list_item.pretty_print()
                                        elif isinstance(list_item, dict) and 'text' in list_item:
                                            print(list_item['text'], end="", flush=True)
                                        else:
                                            print(list_item, end="", flush=True)
                                elif isinstance(v, dict) and 'text' in v:
                                    print(v['text'], end="", flush=True)
                                else:
                                    print(v, end="", flush=True)
                        elif node_chunk is not None:
                            if hasattr(node_chunk, "__iter__") and not isinstance(node_chunk, str):
                                for item in node_chunk:
                                    if isinstance(item, dict) and 'text' in item:
                                        print(item['text'], end="", flush=True)
                                    else:
                                        print(item, end="", flush=True)
                            else:
                                print(node_chunk, end="", flush=True)
                        
                        # 구분선을 여기서 출력하지 않음 (messages 모드와 동일하게)
                        
                    prev_node = node_name
            else:
                # 딕셔너리가 아닌 경우 전체 청크 출력
                print("\n" + "=" * 50)
                print(f"🔄 Raw output 🔄")
                print("- " * 25)
                print(node_chunks, end="", flush=True)
                # 구분선을 여기서 출력하지 않음
                final_result = {"content": node_chunks}

    else:
        raise ValueError(
            f"Invalid stream_mode: {stream_mode}. Must be 'messages' or 'updates'."
        )
    
    # 필요에 따라 최종 결과 반환
    return final_result

async def ainvoke_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: Optional[RunnableConfig] = None,
    node_names: List[str] = [],
    callback: Optional[Callable] = None,
    include_subgraphs: bool = True,
) -> Dict[str, Any]:
    """
    LangGraph 앱의 실행 결과를 비동기적으로 스트리밍하여 출력하는 함수입니다.

    Args:
        graph (CompiledStateGraph): 실행할 컴파일된 LangGraph 객체
        inputs (dict): 그래프에 전달할 입력값 딕셔너리
        config (Optional[RunnableConfig]): 실행 설정 (선택적)
        node_names (List[str], optional): 출력할 노드 이름 목록. 기본값은 빈 리스트
        callback (Optional[Callable], optional): 각 청크 처리를 위한 콜백 함수. 기본값은 None
            콜백 함수는 {"node": str, "content": Any} 형태의 딕셔너리를 인자로 받습니다.
        include_subgraphs (bool, optional): 서브그래프 포함 여부. 기본값은 True

    Returns:
        Dict[str, Any]: 최종 결과 (마지막 노드의 출력)
    """
    config = config or {}
    final_result = {}

    def format_namespace(namespace):
        return namespace[-1].split(":")[0] if len(namespace) > 0 else "root graph"

    # subgraphs 매개변수를 통해 서브그래프의 출력도 포함
    async for chunk in graph.astream(
        inputs, config, stream_mode="updates", subgraphs=include_subgraphs
    ):
        # 반환 형식에 따라 처리 방법 분기
        if isinstance(chunk, tuple) and len(chunk) == 2:
            # 기존 예상 형식: (namespace, chunk_dict)
            namespace, node_chunks = chunk
        else:
            # 단일 딕셔너리만 반환하는 경우 (REACT 에이전트 등)
            namespace = []  # 빈 네임스페이스 (루트 그래프)
            node_chunks = chunk  # chunk 자체가 노드 청크 딕셔너리
        
        # 딕셔너리인지 확인하고 항목 처리
        if isinstance(node_chunks, dict):
            for node_name, node_chunk in node_chunks.items():
                final_result = {"node": node_name, "content": node_chunk, "namespace": namespace}
                
                # node_names가 비어있지 않은 경우에만 필터링
                if node_names and node_name not in node_names:
                    continue

                # 콜백 함수가 있는 경우 실행
                if callback is not None:
                    result = callback({"node": node_name, "content": node_chunk})
                    # 코루틴인 경우 await
                    if hasattr(result, "__await__"):
                        await result
                # 콜백이 없는 경우 기본 출력
                else:
                    print("\n" + "=" * 50)
                    formatted_namespace = format_namespace(namespace)
                    if formatted_namespace == "root graph":
                        print(f"🔄 Node: \033[1;36m{node_name}\033[0m 🔄")
                    else:
                        print(
                            f"🔄 Node: \033[1;36m{node_name}\033[0m in [\033[1;33m{formatted_namespace}\033[0m] 🔄"
                        )
                    print("- " * 25)

                    # 노드의 청크 데이터 출력
                    if isinstance(node_chunk, dict):
                        for k, v in node_chunk.items():
                            if isinstance(v, BaseMessage):
                                v.pretty_print()
                            elif isinstance(v, list):
                                for list_item in v:
                                    if isinstance(list_item, BaseMessage):
                                        list_item.pretty_print()
                                    else:
                                        print(list_item)
                            elif isinstance(v, dict):
                                for node_chunk_key, node_chunk_value in v.items():
                                    print(f"{node_chunk_key}:\n{node_chunk_value}")
                            else:
                                print(f"\033[1;32m{k}\033[0m:\n{v}")
                    elif node_chunk is not None:
                        if hasattr(node_chunk, "__iter__") and not isinstance(node_chunk, str):
                            for item in node_chunk:
                                print(item)
                        else:
                            print(node_chunk)
                    print("=" * 50)
        else:
            # 딕셔너리가 아닌 경우 전체 청크 출력
            print("\n" + "=" * 50)
            print(f"🔄 Raw output 🔄")
            print("- " * 25)
            print(node_chunks)
            print("=" * 50)
            final_result = {"content": node_chunks}
    
    # 최종 결과 반환
    return final_result

FOLLOWUP_PROMPT = """You are an AI assistant that proposes short, helpful follow-up
questions the user might click next. Return 3 suggestions, each on a new line."""

def get_followup_llm(model_name: str,
                     temperature: float,
                     api_key: str,
                     base_url: str):
    """pages 코드에서 쓰는 설정을 그대로 넘겨 받아 ChatOpenAI 인스턴스 생성."""
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=64,
        api_key=api_key,
        base_url=base_url,
    )

async def generate_followups(llm, last_answer: str, k: int = 3) -> list[str]:
    """마지막 assistant 답변으로부터 k개의 follow-up 문장 생성."""
    resp = await llm.ainvoke([
        SystemMessage(content=FOLLOWUP_PROMPT),
        HumanMessage(content=last_answer),
    ])
    return [l.strip("-• ").rstrip()
            for l in resp.content.splitlines()
            if l.strip()][:k]

def tool_callback(tool) -> None:
    print("[도구 호출]")
    print(f"Tool: {tool.get('tool')}")  # 사용된 도구의 이름을 출력합니다.
    if tool_input := tool.get("tool_input"):
        for k, v in tool_input.items():
            print(f"{k}: {v}")
    print(f"Log: {tool.get('log')}")

def observation_callback(observation) -> None:
    print("[관찰 내용]")
    print(f"Observation: {observation.get('observation')}")

def result_callback(result: str) -> None:
    print("[최종 답변]")
    print(result)

@dataclass
class AgentCallbacks:
    """
    에이전트 콜백 함수들을 포함하는 데이터 클래스입니다.

    Attributes:
        tool_callback (Callable[[Dict[str, Any]], None]): 도구 사용 시 호출되는 콜백 함수
        observation_callback (Callable[[Dict[str, Any]], None]): 관찰 결과 처리 시 호출되는 콜백 함수
        result_callback (Callable[[str], None]): 최종 결과 처리 시 호출되는 콜백 함수
    """
    tool_callback: Callable[[Dict[str, Any]], None] = tool_callback
    observation_callback: Callable[[Dict[str, Any]], None] = observation_callback
    result_callback: Callable[[str], None] = result_callback

class AgentStreamParser:
    """
    에이전트의 스트림 출력을 파싱하고 처리하는 클래스입니다.
    """
    def __init__(self, callbacks: AgentCallbacks = AgentCallbacks()):
        self.callbacks = callbacks
        self.output = None

    def process_agent_steps(self, step: Dict[str, Any]) -> None:
        if "actions" in step:
            self._process_actions(step["actions"])
        elif "steps" in step:
            self._process_observations(step["steps"])
        elif "output" in step:
            self._process_result(step["output"])

    def _process_actions(self, actions: List[Any]) -> None:
        for action in actions:
            if isinstance(action, (AgentAction, ToolAgentAction)) and hasattr(
                action, "tool"
            ):
                self._process_tool_call(action)

    def _process_tool_call(self, action: Any) -> None:
        tool_action = {
            "tool": getattr(action, "tool", None),
            "tool_input": getattr(action, "tool_input", None),
            "log": getattr(action, "log", None),
        }
        self.callbacks.tool_callback(tool_action)

    def _process_observations(self, observations: List[Any]) -> None:
        for observation in observations:
            observation_dict = {}
            if isinstance(observation, AgentStep):
                observation_dict["observation"] = getattr(
                    observation, "observation", None
                )
            self.callbacks.observation_callback(observation_dict)

    def _process_result(self, result: str) -> None:
        self.callbacks.result_callback(result)
        self.output = result

def pretty_print_messages(messages: list[BaseMessage]):
    for message in messages:
        message.pretty_print()

# 각 깊이 수준에 대해 미리 정의된 색상 (ANSI 이스케이프 코드 사용)
depth_colors = {
    1: "\033[96m",  # 밝은 청록색 (눈에 잘 띄는 첫 계층)
    2: "\033[93m",  # 노란색 (두 번째 계층)
    3: "\033[94m",  # 밝은 초록색 (세 번째 계층)
    4: "\033[95m",  # 보라색 (네 번째 계층)
    5: "\033[92m",  # 밝은 파란색 (다섯 번째 계층)
    "default": "\033[96m",  # 기본값은 밝은 청록색으로
    "reset": "\033[0m",  # 기본 색상으로 재설정
}

def is_terminal_dict(data):
    """말단 딕셔너리인지 확인합니다."""
    if not isinstance(data, dict):
        return False
    for value in data.values():
        if isinstance(value, (dict, list)) or hasattr(value, "__dict__"):
            return False
    return True

def format_terminal_dict(data):
    """말단 딕셔너리를 포맷팅합니다."""
    items = []
    for key, value in data.items():
        if isinstance(value, str):
            items.append(f'"{key}": "{value}"')
        else:
            items.append(f'"{key}": {value}')
    return "{" + ", ".join(items) + "}"

def _display_message_tree(data, indent=0, node=None, is_root=False):
    """
    JSON 객체의 트리 구조를 타입 정보 없이 출력합니다.
    """
    spacing = " " * indent * 4
    color = depth_colors.get(indent + 1, depth_colors["default"])

    if isinstance(data, dict):
        if not is_root and node is not None:
            if is_terminal_dict(data):
                print(
                    f'{spacing}{color}{node}{depth_colors["reset"]}: {format_terminal_dict(data)}'
                )
            else:
                print(f'{spacing}{color}{node}{depth_colors["reset"]}:')
                for key, value in data.items():
                    _display_message_tree(value, indent + 1, key)
        else:
            for key, value in data.items():
                _display_message_tree(value, indent + 1, key)

    elif isinstance(data, list):
        if not is_root and node is not None:
            print(f'{spacing}{color}{node}{depth_colors["reset"]}:')

        for index, item in enumerate(data):
            print(f'{spacing}    {color}index [{index}]{depth_colors["reset"]}')
            _display_message_tree(item, indent + 1)

    elif hasattr(data, "__dict__") and not is_root:
        if node is not None:
            print(f'{spacing}{color}{node}{depth_colors["reset"]}:')
        _display_message_tree(data.__dict__, indent)

    else:
        if node is not None:
            if isinstance(data, str):
                value_str = f'"{data}"'
            else:
                value_str = str(data)

            print(f'{spacing}{color}{node}{depth_colors["reset"]}: {value_str}')

def display_message_tree(message):
    """
    메시지 트리를 표시하는 주 함수입니다.
    """
    if isinstance(message, BaseMessage):
        _display_message_tree(message.__dict__, is_root=True)
    else:
        _display_message_tree(message, is_root=True)

class ToolChunkHandler:
    """Tool Message 청크를 처리하고 관리하는 클래스"""

    def __init__(self):
        self._reset_state()

    def _reset_state(self) -> None:
        """상태 초기화"""
        self.gathered = None
        self.first = True
        self.current_node = None
        self.current_namespace = None

    def _should_reset(self, node: str | None, namespace: str | None) -> bool:
        """상태 리셋 여부 확인"""
        # 파라미터가 모두 None인 경우 초기화하지 않음
        if node is None and namespace is None:
            return False

        # node만 설정된 경우
        if node is not None and namespace is None:
            return self.current_node != node

        # namespace만 설정된 경우
        if namespace is not None and node is None:
            return self.current_namespace != namespace

        # 둘 다 설정된 경우
        return self.current_node != node or self.current_namespace != namespace

    def process_message(
        self,
        chunk: AIMessageChunk,
        node: str | None = None,
        namespace: str | None = None,
    ) -> None:
        """
        메시지 청크 처리

        Args:
            chunk: 처리할 AI 메시지 청크
            node: 현재 노드명 (선택사항)
            namespace: 현재 네임스페이스 (선택사항)
        """
        if self._should_reset(node, namespace):
            self._reset_state()

        self.current_node = node if node is not None else self.current_node
        self.current_namespace = (
            namespace if namespace is not None else self.current_namespace
        )

        self._accumulate_chunk(chunk)
        return self._display_tool_calls()

    def _accumulate_chunk(self, chunk: AIMessageChunk) -> None:
        """청크 누적"""
        self.gathered = chunk if self.first else self.gathered + chunk
        self.first = False

    def _display_tool_calls(self) -> None:
        """도구 호출 정보 출력"""
        if (
            self.gathered
            and not self.gathered.content
            and self.gathered.tool_call_chunks
            and self.gathered.tool_calls
        ):
            return self.gathered.tool_calls[0]["args"]

def get_role_from_messages(msg):
    if isinstance(msg, HumanMessage):
        return "user"
    elif isinstance(msg, AIMessage):
        return "assistant"
    else:
        return "assistant"

def messages_to_history(messages):
    return "\n".join(
        [f"{get_role_from_messages(msg)}: {msg.content}" for msg in messages]
    )

def stream_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: RunnableConfig,
    node_names: List[str] = [],
    callback: Callable = None,
):
    """
    LangGraph의 실행 결과를 스트리밍하여 출력하는 함수입니다.

    Args:
        graph (CompiledStateGraph): 실행할 컴파일된 LangGraph 객체
        inputs (dict): 그래프에 전달할 입력값 딕셔너리
        config (RunnableConfig): 실행 설정
        node_names (List[str], optional): 출력할 노드 이름 목록. 기본값은 빈 리스트
        callback (Callable, optional): 각 청크 처리를 위한 콜백 함수. 기본값은 None
            콜백 함수는 {"node": str, "content": str} 형태의 딕셔너리를 인자로 받습니다.

    Returns:
        None: 함수는 스트리밍 결과를 출력만 하고 반환값은 없습니다.
    """
    prev_node = ""
    for chunk_msg, metadata in graph.stream(inputs, config, stream_mode="messages"):
        curr_node = metadata["langgraph_node"]

        # node_names가 비어있거나 현재 노드가 node_names에 있는 경우에만 처리
        if not node_names or curr_node in node_names:
            # 콜백 함수가 있는 경우 실행
            if callback:
                callback({"node": curr_node, "content": _extract_chunk_content(chunk_msg).content})
            # 콜백이 없는 경우 기본 출력
            else:
                # 노드가 변경된 경우에만 구분선 출력
                if curr_node != prev_node:
                    print("\n" + "=" * 50)
                    print(f"🔄 Node: \033[1;36m{curr_node}\033[0m 🔄")
                    print("- " * 25)
                print(chunk_msg.content, end="", flush=True)

            prev_node = curr_node

def invoke_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: RunnableConfig,
    node_names: List[str] = [],
    callback: Callable = None,
):
    """
    LangGraph 앱의 실행 결과를 예쁘게 스트리밍하여 출력하는 함수입니다.

    Args:
        graph (CompiledStateGraph): 실행할 컴파일된 LangGraph 객체
        inputs (dict): 그래프에 전달할 입력값 딕셔너리
        config (RunnableConfig): 실행 설정
        node_names (List[str], optional): 출력할 노드 이름 목록. 기본값은 빈 리스트
        callback (Callable, optional): 각 청크 처리를 위한 콜백 함수. 기본값은 None
            콜백 함수는 {"node": str, "content": str} 형태의 딕셔너리를 인자로 받습니다.

    Returns:
        None: 함수는 스트리밍 결과를 출력만 하고 반환값은 없습니다.
    """

    def format_namespace(namespace):
        return namespace[-1].split(":")[0] if len(namespace) > 0 else "root graph"

    # subgraphs=True 를 통해 서브그래프의 출력도 포함
    for namespace, chunk in graph.stream(
        inputs, config, stream_mode="updates", subgraphs=True
    ):
        for node_name, node_chunk in chunk.items():
            # node_names가 비어있지 않은 경우에만 필터링
            if len(node_names) > 0 and node_name not in node_names:
                continue

            # 콜백 함수가 있는 경우 실행
            if callback is not None:
                callback({"node": node_name, "content": node_chunk})
            # 콜백이 없는 경우 기본 출력
            else:
                print("\n" + "=" * 50)
                formatted_namespace = format_namespace(namespace)
                if formatted_namespace == "root graph":
                    print(f"🔄 Node: \033[1;36m{node_name}\033[0m 🔄")
                else:
                    print(
                        f"🔄 Node: \033[1;36m{node_name}\033[0m in [\033[1;33m{formatted_namespace}\033[0m] 🔄"
                    )
                print("- " * 25)

                # 노드의 청크 데이터 출력
                if isinstance(node_chunk, dict):
                    for k, v in node_chunk.items():
                        if isinstance(v, BaseMessage):
                            v.pretty_print()
                        elif isinstance(v, list):
                            for list_item in v:
                                if isinstance(list_item, BaseMessage):
                                    list_item.pretty_print()
                                else:
                                    print(list_item)
                        elif isinstance(v, dict):
                            for node_chunk_key, node_chunk_value in v.items():
                                print(f"{node_chunk_key}:\n{node_chunk_value}")
                        else:
                            print(f"\033[1;32m{k}\033[0m:\n{v}")
                else:
                    if node_chunk is not None:
                        for item in node_chunk:
                            print(item)
                print("=" * 50)

class PandasAgentStreamParser:
    """
    Pandas 에이전트의 스트림 출력을 파싱하고 처리하는 클래스입니다.
    """
    def __init__(self, callbacks: AgentCallbacks = None):
        self.callbacks = callbacks or AgentCallbacks()
        self.output = None

    def process_agent_steps(self, step: Dict[str, Any]) -> None:
        """에이전트 스텝을 처리합니다."""
        if "actions" in step:
            self._process_actions(step["actions"])
        elif "steps" in step:
            self._process_observations(step["steps"])
        elif "output" in step:
            self._process_result(step["output"])
        elif "intermediate_steps" in step:
            self._process_intermediate_steps(step["intermediate_steps"])

    def _process_actions(self, actions: List[Any]) -> None:
        """액션들을 처리합니다."""
        for action in actions:
            if isinstance(action, (AgentAction, ToolAgentAction)) and hasattr(action, "tool"):
                self._process_tool_call(action)

    def _process_tool_call(self, action: Any) -> None:
        """도구 호출을 처리합니다."""
        tool_action = {
            "tool": getattr(action, "tool", None),
            "tool_input": getattr(action, "tool_input", None),
            "log": getattr(action, "log", None),
        }
        self.callbacks.tool_callback(tool_action)

    def _process_observations(self, observations: List[Any]) -> None:
        """관찰 결과들을 처리합니다."""
        for observation in observations:
            observation_dict = {}
            if isinstance(observation, AgentStep):
                observation_dict["observation"] = getattr(observation, "observation", None)
            elif hasattr(observation, "observation"):
                observation_dict["observation"] = getattr(observation, "observation", None)
            else:
                observation_dict["observation"] = str(observation)
            self.callbacks.observation_callback(observation_dict)

    def _process_intermediate_steps(self, intermediate_steps: List[Any]) -> None:
        """중간 스텝들을 처리합니다."""
        for step in intermediate_steps:
            if hasattr(step, "action") and hasattr(step, "observation"):
                # AgentStep 형태
                action = step.action
                observation = step.observation
                
                # 액션 처리
                if hasattr(action, "tool"):
                    tool_action = {
                        "tool": getattr(action, "tool", None),
                        "tool_input": getattr(action, "tool_input", None),
                        "log": getattr(action, "log", None),
                    }
                    self.callbacks.tool_callback(tool_action)
                
                # 관찰 처리
                observation_dict = {"observation": observation}
                self.callbacks.observation_callback(observation_dict)
            elif isinstance(step, tuple) and len(step) == 2:
                # (action, observation) 튜플 형태
                action, observation = step
                
                # 액션 처리
                if hasattr(action, "tool"):
                    tool_action = {
                        "tool": getattr(action, "tool", None),
                        "tool_input": getattr(action, "tool_input", None),
                        "log": getattr(action, "log", None),
                    }
                    self.callbacks.tool_callback(tool_action)
                
                # 관찰 처리
                observation_dict = {"observation": observation}
                self.callbacks.observation_callback(observation_dict)

    def _process_result(self, result: str) -> None:
        """최종 결과를 처리합니다."""
        self.callbacks.result_callback(result)
        self.output = result


def pandas_tool_callback(tool) -> None:
    """Pandas 에이전트용 도구 콜백 함수"""
    print("[🐍 Python 코드 실행]")
    print(f"Tool: {tool.get('tool')}")
    if tool_input := tool.get("tool_input"):
        for k, v in tool_input.items():
            if k == "query":
                print(f"코드:\n{v}")
            else:
                print(f"{k}: {v}")
    if log := tool.get('log'):
        print(f"Log: {log}")


def pandas_observation_callback(observation) -> None:
    """Pandas 에이전트용 관찰 콜백 함수"""
    print("[📊 실행 결과]")
    if "observation" in observation:
        obs = observation["observation"]
        # 결과가 너무 길면 자르기
        if isinstance(obs, str) and len(obs) > 1000:
            obs = obs[:1000] + "..."
        print(f"결과: {obs}")


def pandas_result_callback(result: str) -> None:
    """Pandas 에이전트용 결과 콜백 함수"""
    print("[✅ 최종 답변]")
    print(result)
