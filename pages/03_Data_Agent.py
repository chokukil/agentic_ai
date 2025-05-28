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
from utils import (
    astream_graph, random_uuid, generate_followups, get_followup_llm, 
    PandasAgentStreamParser, AgentCallbacks, pandas_tool_callback, 
    pandas_observation_callback, pandas_result_callback,
    # 추가 import - pandas agent 실시간 처리용
    tool_callback, observation_callback, result_callback,
    AgentStreamParser, ToolChunkHandler, display_message_tree,
    pretty_print_messages, messages_to_history, get_role_from_messages
)
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
import pandas as pd
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from urllib.parse import urlsplit, parse_qs
from utils import generate_followups, get_followup_llm
from langchain_experimental.tools import PythonAstREPLTool
from langchain.agents.agent_types import AgentType
import time

# 🆕 데이터 분석 패키지들 import
import numpy as np
import scipy
import seaborn as sns
import sklearn
from sklearn import datasets, metrics, model_selection, preprocessing, linear_model, ensemble, cluster
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 🔧 비동기 처리를 위한 imports 추가
from langchain.agents import AgentExecutor
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import Tool
from typing import Dict, Any, List, Optional
import functools
import threading

# 🔧 MCP 도구를 pandas agent에서 사용할 수 있도록 동기 호환으로 만드는 함수 (개선)
def make_mcp_tool_sync_compatible(mcp_tool):
    """
    MCP 도구를 pandas agent에서 사용할 수 있도록 동기 호환성을 추가
    pandas agent는 동기적으로 도구를 호출하므로 비동기 MCP 도구들을 래핑해야 함
    """
    try:
        # nest_asyncio가 이미 적용되어 있는지 확인
        nest_asyncio.apply()
        
        def sync_tool_runner(*args, **kwargs):
            """동기 도구 실행 래퍼 (pandas agent용) - 개선된 버전"""
            try:
                # 인자가 dict 형태로 오는 경우 처리
                if len(args) == 1 and isinstance(args[0], dict):
                    tool_input = args[0]
                else:
                    tool_input = kwargs if kwargs else {}
                
                logging.debug(f"Sync tool runner called for {mcp_tool.name} with input: {tool_input}")
                
                # 이미 실행 중인 이벤트 루프가 있는지 확인
                try:
                    loop = asyncio.get_running_loop()
                    logging.debug("Found running event loop, using thread pool")
                    
                    # 새 스레드에서 이벤트 루프 실행
                    result_container = []
                    exception_container = []
                    
                    def run_in_new_thread():
                        try:
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                if hasattr(mcp_tool, 'ainvoke'):
                                    result = new_loop.run_until_complete(mcp_tool.ainvoke(tool_input))
                                elif hasattr(mcp_tool, 'invoke'):
                                    result = mcp_tool.invoke(tool_input)
                                else:
                                    # 함수 호출 방식 시도
                                    result = new_loop.run_until_complete(mcp_tool(tool_input))
                                result_container.append(result)
                            finally:
                                new_loop.close()
                        except Exception as e:
                            logging.error(f"Thread execution error for {mcp_tool.name}: {e}")
                            exception_container.append(e)
                    
                    thread = threading.Thread(target=run_in_new_thread)
                    thread.start()
                    thread.join(timeout=60)  # 60초 타임아웃
                    
                    if exception_container:
                        raise exception_container[0]
                    elif result_container:
                        logging.debug(f"Sync tool runner success for {mcp_tool.name}: {str(result_container[0])[:100]}")
                        return result_container[0]
                    else:
                        raise TimeoutError(f"Tool {mcp_tool.name} execution timed out")
                        
                except RuntimeError:
                    # 실행 중인 루프가 없으면 직접 실행
                    logging.debug("No running event loop, executing directly")
                    if hasattr(mcp_tool, 'ainvoke'):
                        result = asyncio.run(mcp_tool.ainvoke(tool_input))
                    elif hasattr(mcp_tool, 'invoke'):
                        result = mcp_tool.invoke(tool_input)
                    else:
                        result = asyncio.run(mcp_tool(tool_input))
                    
                    logging.debug(f"Direct execution success for {mcp_tool.name}: {str(result)[:100]}")
                    return result
                        
            except Exception as e:
                error_msg = f"Tool execution error for {mcp_tool.name}: {str(e)}"
                logging.error(error_msg)
                return error_msg
        
        # 새로운 동기 호환 도구 생성
        compatible_tool = Tool.from_function(
            sync_tool_runner,
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            infer_schema=True,
        )
        compatible_tool.handle_tool_error = True

        
        # 원본 속성들 복사
        if hasattr(mcp_tool, 'args_schema'):
            compatible_tool.args_schema = mcp_tool.args_schema
        if hasattr(mcp_tool, 'return_direct'):
            compatible_tool.return_direct = mcp_tool.return_direct
            
        logging.debug(f"Made MCP tool {mcp_tool.name} sync compatible")
        return compatible_tool
        
    except Exception as e:
        logging.error(f"Failed to make MCP tool sync compatible: {e}")
        return mcp_tool  # 실패 시 원본 반환

class AsyncPandasAgentWrapper:
    """
    pandas agent를 비동기 방식으로 실행할 수 있도록 래핑하는 클래스 (개선된 버전)
    """
    def __init__(self, pandas_agent):
        self.pandas_agent = pandas_agent
        self.executor = None
        
    async def astream(self, inputs: Dict[str, Any], config: Optional[RunnableConfig] = None):
        """
        pandas agent를 비동기로 스트리밍 실행 - 개선된 버전
        """
        try:
            import asyncio
            import concurrent.futures
            
            if self.executor is None:
                self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            
            # pandas agent의 스트림을 비동기로 실행
            def run_pandas_stream():
                try:
                    logging.debug("Starting pandas agent stream execution")
                    steps = []
                    
                    # pandas agent stream 실행
                    for step in self.pandas_agent.stream(inputs, config=config):
                        logging.debug(f"Pandas agent step: {type(step)} - {step}")
                        steps.append(step)
                        
                    logging.debug(f"Pandas agent completed with {len(steps)} steps")
                    return steps
                    
                except Exception as e:
                    logging.error(f"Pandas agent stream error: {e}")
                    import traceback
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    return [{"error": str(e)}]
            
            # 비동기로 실행
            loop = asyncio.get_event_loop()
            steps = await loop.run_in_executor(self.executor, run_pandas_stream)
            
            # 결과를 하나씩 yield
            for step in steps:
                yield step
                # 작은 지연을 추가하여 실시간 느낌 제공
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logging.error(f"AsyncPandasAgentWrapper.astream error: {e}")
            yield {"error": str(e)}
    
    async def ainvoke(self, inputs: Dict[str, Any], config: Optional[RunnableConfig] = None, **kwargs):
        """
        pandas agent를 비동기로 단일 실행 - 개선된 버전
        """
        try:
            import asyncio
            import concurrent.futures
            
            if self.executor is None:
                self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            
            def run_pandas_invoke():
                try:
                    logging.debug("Starting pandas agent invoke execution")
                    result = self.pandas_agent.invoke(inputs, config=config)
                    logging.debug(f"Pandas agent invoke result: {type(result)} - {str(result)[:200]}")
                    return result
                except Exception as e:
                    logging.error(f"Pandas agent invoke error: {e}")
                    import traceback
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    return {"error": str(e)}
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, run_pandas_invoke)
            return result
            
        except Exception as e:
            logging.error(f"AsyncPandasAgentWrapper.ainvoke error: {e}")
            return {"error": str(e)}

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

# 🆕 데이터 분석 환경 설정 함수
def create_data_analysis_environment(df=None):
    """
    데이터 분석에 필요한 모든 패키지들을 사전에 로드한 환경을 생성합니다.
    plt.show()를 자동으로 Streamlit 호환 버전으로 패치합니다.
    
    Args:
        df: 분석할 DataFrame (선택사항)
    
    Returns:
        dict: 사전 로드된 패키지들과 데이터를 포함한 환경 딕셔너리
    """
    # 🆕 한글 폰트 설정 추가
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import platform
    import warnings
    
    def setup_korean_font():
        """한글 폰트를 설정하는 함수"""
        try:
            # Windows 환경에서 한글 폰트 설정
            if platform.system() == 'Windows':
                # Windows에서 사용 가능한 한글 폰트들 (우선순위 순)
                korean_fonts = ['Malgun Gothic', 'Arial Unicode MS', 'Gulim', 'Dotum', 'Batang']
                
                for font_name in korean_fonts:
                    try:
                        # 폰트가 시스템에 설치되어 있는지 확인
                        available_fonts = [f.name for f in fm.fontManager.ttflist]
                        if font_name in available_fonts:
                            plt.rcParams['font.family'] = font_name
                            plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
                            
                            # 폰트 테스트
                            fig, ax = plt.subplots(figsize=(1, 1))
                            ax.text(0.5, 0.5, '한글테스트', fontsize=10)
                            plt.close(fig)  # 테스트 후 즉시 닫기
                            
                            logging.debug(f"한글 폰트 설정 완료: {font_name}")
                            return True
                    except Exception as e:
                        logging.debug(f"폰트 {font_name} 설정 실패: {e}")
                        continue
                        
            elif platform.system() == 'Darwin':  # macOS
                try:
                    plt.rcParams['font.family'] = 'AppleGothic'
                    plt.rcParams['axes.unicode_minus'] = False
                    logging.debug("한글 폰트 설정 완료: AppleGothic")
                    return True
                except:
                    pass
                    
            elif platform.system() == 'Linux':
                try:
                    # Linux에서 한글 폰트 시도
                    linux_fonts = ['NanumGothic', 'NanumBarunGothic', 'DejaVu Sans']
                    for font_name in linux_fonts:
                        try:
                            plt.rcParams['font.family'] = font_name
                            plt.rcParams['axes.unicode_minus'] = False
                            logging.debug(f"한글 폰트 설정 완료: {font_name}")
                            return True
                        except:
                            continue
                except:
                    pass
            
            # 기본 설정이 실패한 경우
            logging.warning("한글 폰트 설정에 실패했습니다. 기본 폰트를 사용합니다.")
            plt.rcParams['axes.unicode_minus'] = False  # 최소한 마이너스 기호는 보호
            return False
            
        except Exception as e:
            logging.error(f"한글 폰트 설정 중 오류 발생: {e}")
            return False
    
    # 한글 폰트 설정 실행
    setup_korean_font()
    
    # 🆕 폰트 캐시 새로고침 (필요한 경우)
    try:
        # 폰트 캐시가 오래된 경우 새로고침
        fm._rebuild()
    except Exception as e:
        logging.debug(f"폰트 캐시 새로고침 실패 (무시 가능): {e}")
    
    # matplotlib의 원본 show 함수 백업
    original_show = plt.show
    original_clf = plt.clf
    original_cla = plt.cla
    original_close = plt.close
    
    def streamlit_show(*args, **kwargs):
        """
        plt.show()를 Streamlit 환경에서 자동으로 st.pyplot()으로 변환하는 함수
        """
        try:
            # 현재 figure가 있는지 확인
            fig = plt.gcf()
            if fig.get_axes():  # axes가 있으면 실제 플롯이 있다는 의미
                
                # 🆕 현재 진행 중인 메시지에 시각화 추가
                if "current_message_visualizations" not in st.session_state:
                    st.session_state.current_message_visualizations = []
                
                # figure를 base64 이미지로 변환
                import io
                import base64
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='white')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.getvalue()).decode()
                
                # HTML img 태그로 변환 (Streamlit markdown에서 렌더링 가능)
                img_html = f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; height:auto; margin: 10px 0;">'
                
                # 현재 메시지의 시각화 목록에 추가
                st.session_state.current_message_visualizations.append(img_html)
                
                # 🆕 시각화 전용 컨테이너에 표시 (실시간)
                if hasattr(st, '_visualization_container') and st._visualization_container is not None:
                    with st._visualization_container:
                        st.pyplot(fig, clear_figure=False)
                else:
                    # 일반적인 경우
                    st.pyplot(fig, clear_figure=False)
                
                # 🆕 새로운 플롯을 위해 새 figure 생성
                plt.figure()
                
            else:
                # 빈 figure인 경우 원래 show 함수 호출
                original_show(*args, **kwargs)
        except Exception as e:
            # 오류 발생 시 원래 show 함수로 fallback
            print(f"Streamlit show error: {e}")
            original_show(*args, **kwargs)
    
    def protected_clf(*args, **kwargs):
        """plt.clf()를 보호하여 의도치 않은 클리어 방지"""
        # 새 figure를 생성하되 기존 것은 건드리지 않음
        plt.figure()
    
    def protected_cla(*args, **kwargs):
        """plt.cla()를 보호하여 의도치 않은 클리어 방지"""
        # 현재 axes만 클리어하되 figure는 유지
        if plt.gcf().get_axes():
            plt.gca().clear()
    
    def protected_close(*args, **kwargs):
        """plt.close()를 보호하여 표시된 figure는 유지"""
        # 인자가 없으면 현재 figure만 닫기
        if not args and not kwargs:
            plt.figure()  # 새 figure 생성
        else:
            original_close(*args, **kwargs)
    
    # matplotlib show 함수를 패치
    plt.show = streamlit_show
    
    # 🆕 matplotlib 클리어 함수들도 패치하여 의도치 않은 figure 삭제 방지
    plt.clf = protected_clf
    plt.cla = protected_cla  
    plt.close = protected_close
    
    # 추가 시각화 헬퍼 함수들
    def reset_show():
        """원본 matplotlib 함수들로 복원"""
        plt.show = original_show
        plt.clf = original_clf
        plt.cla = original_cla
        plt.close = original_close
    
    def force_show():
        """현재 figure를 강제로 Streamlit에 표시"""
        fig = plt.gcf()
        if fig.get_axes():
            st.pyplot(fig, clear_figure=False)
            # 새로운 figure 생성 (기존 것은 유지)
            plt.figure()
    
    # 🆕 한글 폰트 상태 확인 함수
    def check_korean_font():
        """현재 설정된 폰트와 한글 지원 여부를 확인"""
        current_font = plt.rcParams['font.family']
        unicode_minus = plt.rcParams['axes.unicode_minus']
        
        info = f"""
📝 **폰트 설정 정보:**
- 현재 폰트: {current_font}
- 마이너스 기호 보호: {unicode_minus}
- 플랫폼: {platform.system()}

🎨 **한글 테스트**: 가나다라마바사 ← 이 글자들이 정상적으로 보이면 성공!
"""
        return info
    
    # 🆕 데이터 분석 오류 복구용 헬퍼 함수들
    def safe_dataframe_check(obj):
        """DataFrame을 안전하게 체크하는 함수"""
        if obj is None:
            return False
        if hasattr(obj, 'empty'):
            return not obj.empty
        return bool(obj)
    
    def diagnose_data(df=None):
        """데이터 진단 정보를 반환하는 함수"""
        if df is None and 'df' in locals():
            df = locals()['df']
        if df is None and 'data' in locals():
            df = locals()['data']
        if df is None:
            return "진단할 데이터가 없습니다."
        
        try:
            info = f"""
📊 데이터 진단 결과:
- 크기: {df.shape[0]:,} 행 × {df.shape[1]:,} 열  
- 컬럼: {list(df.columns)}
- 데이터 타입: {dict(df.dtypes)}
- 결측값: {dict(df.isnull().sum())}
- 메모리 사용량: {df.memory_usage(deep=True).sum():,} bytes
"""
            return info
        except Exception as e:
            return f"데이터 진단 중 오류: {str(e)}"
    
    def safe_plot():
        """안전한 플롯 생성을 위한 함수"""
        try:
            fig = plt.gcf()
            if hasattr(fig, 'get_axes') and fig.get_axes():
                st.pyplot(fig, clear_figure=False)
                plt.figure()
                return "플롯이 성공적으로 표시되었습니다."
            else:
                return "표시할 플롯이 없습니다."
        except Exception as e:
            return f"플롯 표시 중 오류: {str(e)}"
    
    # 🆕 시각화 관리용 헬퍼 함수들
    def get_current_visualizations():
        """현재 메시지의 시각화 개수 반환"""
        if "current_message_visualizations" in st.session_state:
            return len(st.session_state.current_message_visualizations)
        return 0
    
    def clear_current_visualizations():
        """현재 메시지의 시각화 데이터 제거"""
        if "current_message_visualizations" in st.session_state:
            count = len(st.session_state.current_message_visualizations)
            st.session_state.current_message_visualizations = []
            return f"{count}개의 시각화 데이터가 제거되었습니다."
        return "제거할 시각화 데이터가 없습니다."
    
    def preview_current_visualizations():
        """현재 메시지의 시각화들을 미리보기"""
        if ("current_message_visualizations" in st.session_state and 
            st.session_state.current_message_visualizations):
            st.write(f"**현재 생성된 시각화 {len(st.session_state.current_message_visualizations)}개:**")
            for i, viz_html in enumerate(st.session_state.current_message_visualizations):
                st.markdown(f"시각화 {i+1}:", unsafe_allow_html=False)
                st.markdown(viz_html, unsafe_allow_html=True)
        else:
            st.write("현재 생성된 시각화가 없습니다.")
    
    analysis_env = {
        # 기본 데이터 분석 패키지들
        "pd": pd,
        "pandas": pd,
        "np": np,
        "numpy": np,
        "scipy": scipy,
        "sns": sns,
        "seaborn": sns,
        "plt": plt,
        "matplotlib": plt,
        
        # Streamlit 
        "st": st,
        
        # scikit-learn 관련
        "sklearn": sklearn,
        "datasets": datasets,
        "metrics": metrics,
        "model_selection": model_selection,
        "preprocessing": preprocessing,
        "linear_model": linear_model,
        "ensemble": ensemble,
        "cluster": cluster,
        
        # 기타 유용한 패키지들
        "warnings": warnings,
        "os": os,
        "sys": sys,
        "json": json,
        "time": time,
        
        # 자주 사용하는 함수들을 직접 접근 가능하게
        "train_test_split": model_selection.train_test_split,
        "StandardScaler": preprocessing.StandardScaler,
        "LinearRegression": linear_model.LinearRegression,
        "RandomForestClassifier": ensemble.RandomForestClassifier,
        "KMeans": cluster.KMeans,
        
        # 🆕 시각화 헬퍼 함수들
        "reset_show": reset_show,
        "force_show": force_show,
        "original_show": original_show,
        "original_clf": original_clf,
        "original_cla": original_cla,
        "original_close": original_close,
        
        # 🆕 폰트 관련 헬퍼 함수들
        "setup_korean_font": setup_korean_font,
        "check_korean_font": check_korean_font,
        
        # 🆕 오류 복구용 헬퍼 함수들
        "safe_dataframe_check": safe_dataframe_check,
        "diagnose_data": diagnose_data,
        "safe_plot": safe_plot,
        
        # 🆕 시각화 관리용 헬퍼 함수들
        "get_current_visualizations": get_current_visualizations,
        "clear_current_visualizations": clear_current_visualizations,
        "preview_current_visualizations": preview_current_visualizations,
    }
    
    # DataFrame이 제공된 경우 추가
    if df is not None:
        analysis_env["df"] = df
        analysis_env["data"] = df  # 일반적인 별명도 추가
    
    return analysis_env

# 🆕 데이터 분석용 PythonAstREPLTool 생성 함수
def create_enhanced_python_tool(df=None):
    """
    데이터 분석 패키지들이 사전 로드된 PythonAstREPLTool을 생성합니다.
    plt.show()가 자동으로 Streamlit에서 동작하도록 패치되어 있습니다.
    
    Args:
        df: 분석할 DataFrame (선택사항)
    
    Returns:
        PythonAstREPLTool: 향상된 Python REPL 도구
    """
    analysis_env = create_data_analysis_environment(df)
    
    # 사용자 친화적인 설명과 예제 추가
    description = """
    🤖 **지능형 데이터 분석 환경**에 오신 것을 환영합니다!
    
    📊 **사전 로드된 패키지들:**
    - 데이터 처리: pandas (pd), numpy (np)
    - 시각화: matplotlib (plt), seaborn (sns), streamlit (st)  
    - 머신러닝: scikit-learn (sklearn)
    - 과학계산: scipy
    
    🚀 **특별 기능들:**
    ✅ CSV 업로드 시 **수동 EDA(탐색적 데이터 분석)** 버튼 제공
    ✅ **데이터 특성 자동 분석** 및 **분석 방향 추천**  
    ✅ plt.show() 자동 Streamlit 변환 (시각화 영구 보존)
    ✅ **맞춤형 후속 질문** 자동 생성
    ✅ plt.clf(), plt.cla(), plt.close() 등 클리어 함수들로부터 보호
    ✅ 도구 호출 정보가 접혀도 시각화는 그대로 유지!
    
    📈 **자동 분석 항목:**
    - 데이터 크기, 타입, 결측값 현황
    - 수치형/범주형/날짜형 컬럼 분류  
    - 기본 통계 요약 및 분포 시각화
    - 분석 우선순위 및 방향 제안
    
    🎯 **시작 방법:**
    1. CSV 파일 업로드 → Agent 생성
    2. 사이드바의 '🚀 자동 데이터 분석 시작' 버튼 클릭
    3. 제안된 분석 중 원하는 것 선택
    4. 대화형으로 심화 분석 진행
    
    💬 **사용 예시:**
    - "결측값을 처리해줘"
    - "상관관계를 시각화해줘" 
    - "이상치를 찾아줘"
    - "클러스터링 분석을 해줘"
    
    DataFrame은 'df' 또는 'data' 변수로 접근할 수 있습니다.
    무엇을 분석하고 싶으신지 말씀해 주세요! 🤖✨
    """
    
    return PythonAstREPLTool(
        locals=analysis_env,
        description=description,
        name="enhanced_python_repl",
        handle_tool_error=True
    )

# 🆕 자동 데이터 분석 및 인사말 생성 함수
def auto_analyze_and_greet(df):
    """
    데이터 로드 시 자동으로 기본 분석을 수행하고 인사말과 가이드를 생성합니다.
    """
    try:
        # 데이터 기본 정보 수집
        shape = df.shape
        columns = df.columns.tolist()
        dtypes = df.dtypes.value_counts().to_dict()
        missing_values = df.isnull().sum().sum()
        missing_cols = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
        
        # 수치형/범주형 컬럼 분류
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # 메모리 사용량
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        # 인사말 및 분석 결과 생성
        greeting_content = f"""🎉 **데이터 분석 환경에 오신 것을 환영합니다!**

📊 **로드된 데이터 개요:**
- **데이터 크기**: {shape[0]:,} 행 × {shape[1]:,} 열
- **메모리 사용량**: {memory_usage:.2f} MB
- **결측값**: {missing_values:,} 개 ({missing_values/df.size*100:.1f}%)

📋 **컬럼 구성:**
- **수치형 컬럼** ({len(numeric_cols)}개): {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}
- **범주형 컬럼** ({len(categorical_cols)}개): {', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''}
{'- **날짜형 컬럼** (' + str(len(datetime_cols)) + '개): ' + ', '.join(datetime_cols[:3]) + ('...' if len(datetime_cols) > 3 else '') if datetime_cols else ''}

🔍 **데이터 미리보기:**"""

        # 데이터 미리보기를 텍스트로 포함
        preview_text = df.head(3).to_string()
        greeting_content += f"\n```\n{preview_text}\n```\n"
        
        # 분석 제안 생성
        suggestions = []
        
        if missing_values > 0:
            suggestions.append(f"📍 **결측값 처리**: {len(missing_cols)}개 컬럼에 결측값이 있습니다")
            
        if len(numeric_cols) >= 2:
            suggestions.append("📈 **상관관계 분석**: 수치형 변수들 간의 상관관계를 확인해보세요")
            
        if len(categorical_cols) > 0:
            suggestions.append("📊 **범주형 데이터 분포**: 카테고리별 빈도와 분포를 살펴보세요")
            
        if len(numeric_cols) > 0:
            suggestions.append("📉 **기초 통계**: 수치형 데이터의 분포와 이상치를 확인해보세요")
            
        if shape[0] > 1000:
            suggestions.append("🎯 **샘플링**: 큰 데이터셋이므로 샘플링을 고려해보세요")
        
        # followups를 suggestions에서 동적으로 생성
        followups = [
            s.split(":")[0]
            .replace("📍", "")
            .replace("📈", "")
            .replace("📊", "")
            .replace("📉", "")
            .replace("🎯", "")
            .replace("**", "")
            .strip()
            for s in suggestions
        ]
        if not followups:
            followups = ["데이터의 기본 정보를 보여줘", "첫 5행을 보여줘", "데이터 요약 통계를 보여줘"]

        # 구체적인 분석 명령어 제안
        greeting_content += """
🚀 **빠른 시작 명령어:**
- `df.describe()` - 기초 통계 요약
- `df.info()` - 데이터 타입 및 결측값 정보  
- `df.hist(figsize=(12, 8)); plt.show()` - 전체 변수 히스토그램
- `sns.heatmap(df.corr(), annot=True); plt.show()` - 상관관계 히트맵
- `df.isnull().sum()` - 결측값 확인

무엇을 분석하고 싶으신지 말씀해 주세요! 🤖✨"""

        # 추천 분석 단계 블록을 맨 마지막에 추가
        greeting_content += "\n\n💡 **추천 분석 단계:**\n"
        for i, suggestion in enumerate(suggestions[:4], 1):
            greeting_content += f"{i}. {suggestion}\n"
        
        # 시각화 생성을 위한 기본 플롯
        visualizations = []
        try:
            # 간단한 데이터 개요 시각화 생성
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 1. 데이터 타입 분포 파이차트
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 컬럼 타입 분포
            type_counts = {'수치형': len(numeric_cols), '범주형': len(categorical_cols), '날짜형': len(datetime_cols)}
            type_counts = {k: v for k, v in type_counts.items() if v > 0}
            
            ax1.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%', startangle=90)
            ax1.set_title('컬럼 타입 분포')
            
            # 결측값 현황
            if missing_values > 0 and len(missing_cols) <= 10:
                missing_data = pd.Series(missing_cols)
                missing_data.plot(kind='bar', ax=ax2, color='coral')
                ax2.set_title('컬럼별 결측값 개수')
                ax2.set_ylabel('결측값 개수')
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            else:
                ax2.text(0.5, 0.5, f'전체 결측값: {missing_values:,}개\n({missing_values/df.size*100:.1f}%)', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title('결측값 현황')
                ax2.axis('off')
            
            plt.tight_layout()
            
            # figure를 base64로 변환하여 저장
            import io
            import base64
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='white')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode()
            img_html = f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; height:auto; margin: 10px 0;">'
            
            visualizations.append(img_html)
            plt.close(fig)  # 메모리 정리
            
        except Exception as viz_error:
            logging.warning(f"초기 시각화 생성 실패: {viz_error}")
        
        # 분석 결과를 세션 상태에 저장 (history에는 추가하지 않음)
        analysis_result = {
            "content": greeting_content,
            "visualizations": visualizations,
            "followups": followups
        }
        
        # 🆕 분석 결과를 별도 저장 (초기화 완료 후 사용)
        st.session_state.auto_analysis_result = analysis_result
        
        return True
        
    except Exception as e:
        logging.error(f"자동 데이터 분석 중 오류: {e}")
        # 간단한 인사말만 저장
        simple_greeting = f"""🎉 **데이터 분석 환경에 오신 것을 환영합니다!**

📊 **데이터가 성공적으로 로드되었습니다**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열

데이터를 분석하고 싶은 내용을 말씀해 주세요! 🤖✨"""
        
        st.session_state.auto_analysis_result = {
            "content": simple_greeting,
            "visualizations": [],
            "followups": ["데이터의 기본 정보를 보여줘", "첫 5행을 보여줘", "데이터 요약 통계를 보여줘"]
        }
        
        return False

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
            
            # Create agent based on whether DataFrame is available
            if tool_count > 0 or st.session_state.dataframe is not None:
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
                    streaming=True,  # Enable streaming
                )
                try:
                    # --- 에이전트 생성 ---
                    if st.session_state.dataframe is not None:          # DataFrame이 있으면
                        df = st.session_state.dataframe
                        
                        # 🆕 데이터 분석용 도구 생성
                        enhanced_python_tool = create_enhanced_python_tool(df)
                        
                        # 🔧 수정: MCP 도구들을 pandas agent용 동기 호환으로 변환
                        extra_tools = [enhanced_python_tool]
                        if tools:
                            # MCP 도구들을 동기 호환으로 변환
                            sync_compatible_tools = []
                            for tool in tools:
                                try:
                                    sync_tool = make_mcp_tool_sync_compatible(tool)
                                    sync_compatible_tools.append(sync_tool)
                                    logging.debug(f"Successfully made {tool.name} sync compatible")
                                except Exception as e:
                                    logging.error(f"Failed to make {tool.name} sync compatible: {e}")
                                    # 실패한 도구는 건너뛰기
                                    continue
                            
                            extra_tools.extend(sync_compatible_tools)
                            logging.debug(f"Adding {len(sync_compatible_tools)} sync-compatible MCP tools to pandas agent")
                        
                        # Create pandas agent with sync-compatible MCP tools
                        pandas_agent = create_pandas_dataframe_agent(
                            model,
                            df,
                            verbose=True,
                            agent_type=AgentType.OPENAI_FUNCTIONS,
                            allow_dangerous_code=True,
                            prefix=st.session_state.selected_prompt_text,
                            handle_parsing_errors=True,
                            max_iterations=10,
                            early_stopping_method="generate",
                            extra_tools=extra_tools  # 동기 호환 MCP 도구들 추가
                        )
                        
                        # 🔧 비동기 래퍼로 감싸기
                        async_pandas_wrapper = AsyncPandasAgentWrapper(pandas_agent)
                        
                        st.session_state.agent = async_pandas_wrapper
                        st.session_state.agent_type = "pandas"
                        logging.debug(f'Enhanced async pandas agent with {len(extra_tools)} total tools created successfully')
                        
                        # 🆕 사용 가능한 패키지 정보를 사용자에게 표시
                        st.sidebar.success("✅ 지능형 데이터 분석 환경 준비 완료!")
                        
                        with st.sidebar.expander("📦 사전 로드된 패키지", expanded=False):
                            st.write(f"""
                            **데이터 처리:**
                            - pandas (pd), numpy (np)
                            
                            **시각화:**
                            - matplotlib (plt), seaborn (sns)
                            - ✨ plt.show() 자동 Streamlit 변환 (영구 보존)
                            
                            **머신러닝:**
                            - scikit-learn (sklearn)
                            - datasets, metrics, model_selection
                            - preprocessing, linear_model, ensemble, cluster
                            
                            **과학계산:**
                            - scipy
                            
                            **MCP 도구 (동기 호환):**
                            - {len(tools)}개의 외부 도구 통합 (pandas agent 호환)
                            
                            **추천 시작 명령어:**
                            - `df.describe()` - 기초 통계 요약
                            - `df.hist(); plt.show()` - 히스토그램
                            - `sns.heatmap(df.corr()); plt.show()` - 상관관계
                            """)
                        
                    else:                                               # DataFrame이 없으면 기존 ReAct 유지
                        # 🆕 일반 에이전트에도 향상된 Python 도구 추가
                        enhanced_tools = tools.copy() if tools else []
                        enhanced_python_tool = create_enhanced_python_tool()
                        enhanced_tools.append(enhanced_python_tool)
                        if tools:
                            sync_compatible_tools = []
                            for tool in tools:
                                try:
                                    sync_tool = make_mcp_tool_sync_compatible(tool)
                                    sync_compatible_tools.append(sync_tool)
                                    logging.debug(f"Successfully made {tool.name} sync compatible")
                                except Exception as e:
                                    logging.error(f"Failed to make {tool.name} sync compatible: {e}")
                                    continue
                            enhanced_tools.extend(sync_compatible_tools)
                        
                        agent = create_react_agent(
                            model,
                            enhanced_tools,
                            prompt=st.session_state.selected_prompt_text,
                            checkpointer=MemorySaver(),
                        )
                        st.session_state.agent = agent
                        st.session_state.agent_type = "langgraph"
                        logging.debug('Enhanced LangGraph ReAct agent created successfully')
                        
                except Exception as e:
                    logging.error(f"Error creating agent: {str(e)}")
                    import traceback
                    logging.error(f"Agent creation error details:\n{traceback.format_exc()}")
                    st.error(f"에이전트 생성 중 오류가 발생했습니다: {str(e)}")
                    st.session_state.agent = None
                    st.session_state.agent_type = None
            else:
                st.session_state.agent = None
                st.session_state.agent_type = None
                logging.warning('No tools available and no DataFrame loaded, agent not created.')
            
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
                # 🆕 메시지 내용을 HTML로 렌더링 (시각화 포함)
                content = message["content"]
                
                # 시각화가 포함된 경우 HTML로 렌더링
                if "visualizations" in message and message["visualizations"]:
                    # 텍스트 내용 먼저 표시
                    if content and content.strip():
                        st.write(content)
                    
                    # 시각화들을 HTML로 표시
                    for viz_html in message["visualizations"]:
                        st.markdown(viz_html, unsafe_allow_html=True)
                else:
                    # 일반 텍스트만 있는 경우
                    st.write(content)

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
    스트리밍 콜백 함수를 생성합니다. (개선된 버전)

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
    
    # 🆕 pandas agent용 스트림 파서 초기화 (utils.py에서 가져온 함수들 사용)
    pandas_callbacks = AgentCallbacks(
        tool_callback=lambda tool: _handle_pandas_tool_display(tool, accumulated_tool, tool_placeholder),
        observation_callback=lambda obs: _handle_pandas_observation_display(obs, accumulated_tool, tool_placeholder),
        result_callback=lambda result: _handle_pandas_result_display(result, accumulated_text, text_placeholder)
    )
    pandas_parser = PandasAgentStreamParser(pandas_callbacks)

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
        
        # Extract node and content from the message
        node = message.get("node", "")
        
        # 🆕 pandas agent streaming 처리 - utils.py의 PandasAgentStreamParser 사용
        if isinstance(message_content, dict) and {"actions", "steps"} & message_content.keys():
            try:
                pandas_parser.process_agent_steps(message_content)
                return None
            except Exception as e:
                logging.warning(f"Error in pandas agent stream parser: {e}")
                # Fallback to original processing
        
        # Handle pandas agent streaming - process step-by-step execution (fallback)
        if isinstance(message_content, dict):
            # Handle pandas agent steps
            if "actions" in message_content:
                actions = message_content["actions"]
                for action in actions:
                    tool_name = getattr(action, "tool", "Unknown Tool")
                    tool_input = getattr(action, "tool_input", {})
                    
                    # Display tool execution
                    if tool_name == "python_repl_ast" or tool_name == "enhanced_python_repl":  # 🆕 향상된 도구 이름도 체크
                        query = tool_input.get("query", "")
                        entry = f"\n**🐍 Python 코드 실행 (Enhanced):**\n```python\n{query}\n```\n"
                        accumulated_tool.append(entry)
                        tool_placeholder.markdown("".join(accumulated_tool))
                        time.sleep(0.01)
                        logging.debug(f"Added Enhanced Python code execution: {query[:100]}...")
                    else:
                        entry = f"\n**도구 호출: {tool_name}**\n입력: {json.dumps(tool_input, indent=2, ensure_ascii=False)}\n"
                        accumulated_tool.append(entry)
                        tool_placeholder.markdown("".join(accumulated_tool))
                        time.sleep(0.01)
                        logging.debug(f"Added tool call: {tool_name}")
                return None
                
            # Handle pandas agent observations
            if "steps" in message_content:
                steps = message_content["steps"]
                for step in steps:
                    obs = getattr(step, "observation", None)
                    # 🆕 DataFrame 호환성을 위한 안전한 None 체크
                    if obs is not None and not (hasattr(obs, 'empty') and obs.empty):
                        # Format observation output
                        obs_str = str(obs)
                        if len(obs_str) > 1000:
                            obs_str = obs_str[:1000] + "..."
                        entry = f"\n**📊 실행 결과:**\n```\n{obs_str}\n```\n"
                        accumulated_tool.append(entry)
                        tool_placeholder.markdown("".join(accumulated_tool))
                        time.sleep(0.01)
                        logging.debug(f"Added observation result: {obs_str[:100]}...")
                return None
                
            # Handle final output from pandas agent
            if "output" in message_content:
                output = message_content["output"]
                if isinstance(output, str) and output.strip():
                    accumulated_text.append(output)
                    text_placeholder.markdown("".join(accumulated_text))
                    time.sleep(0.01)
                    logging.debug(f"Added final output: {output[:100]}...")
                return None
                
            # Handle intermediate_steps for pandas agent
            if "intermediate_steps" in message_content:
                steps = message_content["intermediate_steps"]
                for step in steps:
                    if hasattr(step, "action") and hasattr(step, "observation"):
                        # AgentStep 형태
                        action = step.action
                        observation = step.observation
                        
                        # 액션 처리
                        if hasattr(action, "tool"):
                            tool_name = getattr(action, "tool", "Unknown Tool")
                            tool_input = getattr(action, "tool_input", {})
                            
                            if tool_name == "python_repl_ast" or tool_name == "enhanced_python_repl":  # 🆕
                                query = tool_input.get("query", "")
                                entry = f"\n**🐍 Python 코드 실행 (Enhanced):**\n```python\n{query}\n```\n"
                            else:
                                entry = f"\n**도구 호출: {tool_name}**\n입력: {json.dumps(tool_input, indent=2, ensure_ascii=False)}\n"
                            accumulated_tool.append(entry)
                        
                        # 관찰 처리 - 🆕 DataFrame 호환성을 위한 안전한 체크
                        if observation is not None and not (hasattr(observation, 'empty') and observation.empty):
                            obs_str = str(observation)
                            if len(obs_str) > 1000:
                                obs_str = obs_str[:1000] + "..."
                            entry = f"\n**📊 실행 결과:**\n```\n{obs_str}\n```\n"
                            accumulated_tool.append(entry)
                    elif isinstance(step, tuple) and len(step) == 2:
                        # (action, observation) 튜플 형태
                        action, observation = step
                        
                        if hasattr(action, "tool"):
                            tool_name = getattr(action, "tool", "Unknown Tool")
                            tool_input = getattr(action, "tool_input", {})
                            
                            if tool_name == "python_repl_ast" or tool_name == "enhanced_python_repl":  # 🆕
                                query = tool_input.get("query", "")
                                entry = f"\n**🐍 Python 코드 실행 (Enhanced):**\n```python\n{query}\n```\n"
                            else:
                                entry = f"\n**도구 호출: {tool_name}**\n입력: {json.dumps(tool_input, indent=2, ensure_ascii=False)}\n"
                            accumulated_tool.append(entry)
                        
                        # 관찰 처리 - 🆕 DataFrame 호환성을 위한 안전한 체크  
                        if observation is not None and not (hasattr(observation, 'empty') and observation.empty):
                            obs_str = str(observation)
                            if len(obs_str) > 1000:
                                obs_str = obs_str[:1000] + "..."
                            entry = f"\n**📊 실행 결과:**\n```\n{obs_str}\n```\n"
                            accumulated_tool.append(entry)
                
                tool_placeholder.markdown("".join(accumulated_tool))
                time.sleep(0.01)
                return None
        
        # Handle different node types from AgentExecutor
        if node == "llm" and message_content:
            # LLM streaming chunks
            logging.debug(f"Processing LLM node with content: {message_content}")
            if hasattr(message_content, "content") and message_content.content:
                content = message_content.content
                if isinstance(content, str) and content.strip():
                    accumulated_text.append(content)
                    # Use markdown for better real-time display
                    text_placeholder.markdown("".join(accumulated_text))
                    # Force UI update
                    time.sleep(0.01)
                    logging.debug(f"Added LLM content to text: {content[:100]}...")
            return None
            
        elif node == "tool" and message_content:
            # Tool execution info
            logging.debug(f"Processing tool node with content: {message_content}")
            if isinstance(message_content, dict):
                if "tool" in message_content and "tool_input" in message_content:
                    # Tool start
                    tool_name = message_content["tool"]
                    tool_input = message_content["tool_input"]
                    entry = f"\n**도구 호출: {tool_name}**\n입력: {json.dumps(tool_input, indent=2, ensure_ascii=False)}\n"
                    accumulated_tool.append(entry)
                    tool_placeholder.markdown("".join(accumulated_tool))
                    time.sleep(0.01)
                    logging.debug(f"Added tool start info: {tool_name}")
                elif "tool" in message_content and "output" in message_content:
                    # Tool end
                    tool_name = message_content["tool"]
                    tool_output = message_content["output"]
                    entry = f"\n**도구 결과: {tool_name}**\n{str(tool_output)[:1000]}...\n"
                    accumulated_tool.append(entry)
                    tool_placeholder.markdown("".join(accumulated_tool))
                    time.sleep(0.01)
                    logging.debug(f"Added tool result: {tool_name}")
            return None
            
        elif node == "agent" and message_content:
            # Final agent output
            logging.debug(f"Processing agent node with content: {message_content}")
            if isinstance(message_content, dict):
                output_text = message_content.get("output", "")
                if isinstance(output_text, str) and output_text.strip():
                    accumulated_text.append(output_text)
                    text_placeholder.markdown("".join(accumulated_text))
                    time.sleep(0.01)
                    logging.debug(f"Added agent output to text: {output_text[:100]}...")
            return None

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
                            if tool_name=='execute_python' or tool_name == "enhanced_python_repl":  # 🆕
                                entry = f"\n**도구 호출: {tool_name}**\n\n{raw_display}\n"
                            else:
                                entry = f"\n**도구 호출: {tool_name}**\n\n{raw_display}\n"
                        else:
                            entry = raw_display
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
                entry = tool_str.replace("\n", "")
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
                        else:
                            entry = raw_display.replace("\n", "")
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
        
        # Fallback: handle original message format for backward compatibility
        elif message_content:
            logging.debug(f"Fallback processing - Message content type: {type(message_content)}")
            if isinstance(message_content, str):
                # actions/steps/output 등 키워드만 들어오면 무시
                if message_content.strip() in {"output", "actions", "steps", "intermediate_steps"}:
                    logging.debug(f"Ignored string '{message_content.strip()}' in streaming callback")
                    return None
                # 그 외의 str만 누적(실제 답변 텍스트가 str로 올 때만)
                accumulated_text.append(message_content)
                text_placeholder.markdown("".join(accumulated_text))
                logging.debug(f"Added string content to text: {message_content[:100]}...")
                return None
        else:
            logging.debug("No message content found")
        
        return None

    return callback_func, accumulated_text, accumulated_tool


def _handle_pandas_tool_display(tool, accumulated_tool, tool_placeholder):
    """pandas agent 도구 호출 시 Streamlit 표시 처리"""
    tool_name = tool.get('tool', 'Unknown Tool')
    tool_input = tool.get('tool_input', {})
    
    if tool_name == "python_repl_ast" or tool_name == "enhanced_python_repl":
        query = tool_input.get("query", "")
        entry = f"\n**🐍 Python 코드 실행:**\n```python\n{query}\n```\n"
    else:
        entry = f"\n**🔧 도구 호출: {tool_name}**\n입력: {json.dumps(tool_input, indent=2, ensure_ascii=False)}\n"
    
    accumulated_tool.append(entry)
    tool_placeholder.markdown("".join(accumulated_tool))
    time.sleep(0.01)


def _handle_pandas_observation_display(observation, accumulated_tool, tool_placeholder):
    """pandas agent 관찰 결과 시 Streamlit 표시 처리"""
    obs = observation.get("observation")
    if obs is not None and not (hasattr(obs, 'empty') and obs.empty):
        obs_str = str(obs)
        if len(obs_str) > 1000:
            obs_str = obs_str[:1000] + "..."
        entry = f"\n**📊 실행 결과:**\n```\n{obs_str}\n```\n"
        accumulated_tool.append(entry)
        tool_placeholder.markdown("".join(accumulated_tool))
        time.sleep(0.01)


def _handle_pandas_result_display(result, accumulated_text, text_placeholder):
    """pandas agent 최종 결과 시 Streamlit 표시 처리"""
    if isinstance(result, str) and result.strip():
        accumulated_text.append(result)
        text_placeholder.markdown("".join(accumulated_text))
        time.sleep(0.01)


# Handle tool execution for sync-compatible tools (개선된 버전)
async def execute_tool_sync_compatible(tool_call, tools):
    """Execute a sync-compatible tool and return its response - 개선된 버전"""
    tool_id = tool_call.get("id")
    tool_name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})
    
    logging.debug(f"Executing sync-compatible tool: {tool_name} (ID: {tool_id})")
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
        # Execute the sync-compatible tool
        # 동기 호환 도구는 .run() 메서드를 사용 (Tool 클래스)
        if hasattr(matching_tool, 'run'):
            result = matching_tool.run(arguments)
        elif hasattr(matching_tool, 'invoke'):
            result = matching_tool.invoke(arguments)
        elif hasattr(matching_tool, 'func'):
            # Tool 클래스의 func 속성을 직접 호출
            result = matching_tool.func(arguments)
        else:
            # Fallback to async invoke if available
            result = await matching_tool.ainvoke(arguments)
            
        logging.debug(f"Sync-compatible tool execution result: {str(result)[:200]}...")
        
        # Create response
        return {
            "tool_call_id": tool_id,
            "name": tool_name,
            "content": str(result)
        }
    except Exception as e:
        import traceback
        error_msg = f"Error executing sync-compatible tool {tool_name}: {str(e)}"
        error_trace = traceback.format_exc()
        logging.error(f"{error_msg}\n{error_trace}")
        return {
            "tool_call_id": tool_id,
            "name": tool_name,
            "content": f"Error: {error_msg}"
        }


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
    pandas agent와 langgraph agent 모두 비동기로 처리합니다. (개선된 버전)
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
            
            # 🆕 새 메시지를 위해 시각화 데이터 초기화
            st.session_state.current_message_visualizations = []
            
            try:
                logging.debug(f"Agent type: {type(st.session_state.agent)}")
                
                # Check if this is a pandas agent or regular LangGraph agent
                agent_type = st.session_state.get("agent_type", "unknown")
                
                if agent_type == "pandas":
                    # 🔧 수정: pandas agent를 비동기 방식으로 처리 (개선된 버전)
                    logging.debug("Processing pandas agent query using async wrapper")
                    
                    try:
                        # Clear the progress message
                        text_placeholder.markdown("")
                        
                        config = RunnableConfig(
                            recursion_limit=st.session_state.recursion_limit,
                            thread_id=st.session_state.thread_id,
                            configurable={
                                "callbacks": [
                                    lambda x: logging.debug(f"RunnableConfig callback: {str(x)[:100]}...")
                                ]
                            }
                        )
                        
                        logging.debug(f"Starting async pandas agent execution with timeout: {timeout_seconds}s")
                        
                        # 🔧 비동기 래퍼를 사용하여 스트리밍 (개선된 버전)
                        final_output = ""
                        error_occurred = False
                        error_message = ""
                        step_count = 0
                        
                        # 비동기 스트리밍 처리 (실시간 업데이트)
                        try:
                            async for step in st.session_state.agent.astream(
                                {"input": query}, 
                                config=config
                            ):
                                step_count += 1
                                logging.debug(f"Async pandas agent step {step_count}: {type(step)} - {list(step.keys()) if isinstance(step, dict) else step}")
                                
                                try:
                                    # Process the step using our callback
                                    streaming_callback({"node": "pandas_agent", "content": step})
                                    
                                    # Extract final output if available
                                    if isinstance(step, dict) and "output" in step:
                                        final_output = step["output"]
                                        logging.debug(f"Found final output: {str(final_output)[:100]}...")
                                        
                                    # 실시간 UI 업데이트를 위한 작은 지연
                                    await asyncio.sleep(0.01)
                                        
                                except Exception as step_error:
                                    error_occurred = True
                                    error_message = str(step_error)
                                    logging.warning(f"Error processing async pandas agent step: {error_message}")
                                    
                                    error_entry = f"\n**⚠️ 처리 중 오류 발생:**\n```\n{error_message}\n```\n"
                                    tool_placeholder.markdown(error_entry)
                                    continue
                        
                        except Exception as stream_error:
                            # 스트리밍이 실패한 경우 직접 invoke 시도
                            logging.warning(f"Async streaming failed, trying direct invoke: {stream_error}")
                            try:
                                result = await st.session_state.agent.ainvoke(
                                    {"input": query}, 
                                    config=config
                                )
                                if isinstance(result, dict) and "output" in result:
                                    final_output = result["output"]
                                else:
                                    final_output = str(result)
                                    
                                # 직접 실행 결과를 스트리밍 콜백으로 처리
                                streaming_callback({"node": "pandas_agent", "content": {"output": final_output}})
                                
                            except Exception as invoke_error:
                                error_occurred = True
                                error_message = f"Both streaming and direct invoke failed: {invoke_error}"
                                logging.error(error_message)
                        
                        # Handle tool calls if any were detected during streaming
                        if "pending_tool_calls" in st.session_state and st.session_state.pending_tool_calls:
                            logging.debug(f"Processing {len(st.session_state.pending_tool_calls)} pending tool calls")
                            
                            # Process each pending tool call
                            while st.session_state.pending_tool_calls:
                                tool_call = st.session_state.pending_tool_calls[0]
                                logging.debug(f"Processing tool call: {tool_call}")
                                
                                # 🔧 수정: pandas agent용 동기 호환 도구들 사용 (개선된 버전)
                                available_tools = []
                                if st.session_state.mcp_client:
                                    # MCP 클라이언트에서 도구 가져오기
                                    mcp_tools = st.session_state.mcp_client.get_tools()
                                    # pandas agent에서는 동기 호환 도구들로 변환
                                    for tool in mcp_tools:
                                        try:
                                            sync_tool = make_mcp_tool_sync_compatible(tool)
                                            available_tools.append(sync_tool)
                                        except Exception as e:
                                            logging.error(f"Failed to make tool {tool.name} sync compatible: {e}")
                                            continue
                                
                                # 동기 호환 도구로 실행하되 비동기 인터페이스 유지
                                tool_result = await execute_tool_sync_compatible(tool_call, available_tools)
                                
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
                                
                                # Remove processed tool call
                                st.session_state.pending_tool_calls = st.session_state.pending_tool_calls[1:]
                        
                        # 최종 출력 처리 (개선된 버전)
                        if error_occurred and final_output:
                            final_output += f"\n\n⚠️ 처리 중 일부 오류가 발생했지만 결과를 생성했습니다: {error_message}"
                            
                        elif error_occurred and not final_output:
                            final_output = f"❌ 처리 중 오류가 발생했습니다: {error_message}\n\n💡 다음을 시도해보세요:\n- 더 구체적인 질문으로 다시 시도\n- 데이터 형태나 컬럼명 확인\n- 단계별로 나누어 질문"
                            
                        # Ensure final text is displayed
                        if final_output and not accumulated_text_obj:
                            accumulated_text_obj.append(final_output)
                            text_placeholder.markdown("".join(accumulated_text_obj))
                        
                        response = {"output": final_output}
                        logging.debug(f"Pandas agent processing completed. Steps: {step_count}, Final output length: {len(final_output)}")
                        
                    except Exception as e:
                        import traceback
                        error_msg = f"Async pandas agent 처리 중 오류 발생: {str(e)}"
                        error_trace = traceback.format_exc()
                        logging.error(f"{error_msg}\n{error_trace}")
                        
                        # 사용자에게 도움이 되는 정보 제공 (개선된 버전)
                        user_friendly_error = f"""❌ 데이터 분석 중 오류가 발생했습니다.

**오류 내용:** {str(e)}

💡 **해결 방법:**
1. **데이터 확인**: `df.head()`, `df.info()`, `df.describe()` 로 데이터 상태 확인
2. **컬럼명 확인**: `df.columns.tolist()` 로 정확한 컬럼명 확인  
3. **단계별 접근**: 복잡한 분석을 단계별로 나누어 수행
4. **구체적 질문**: "특정 컬럼의 평균값은?" 같이 구체적으로 질문

**재시도 예시:**
- "데이터의 기본 정보를 보여줘"
- "첫 5행을 보여줘" 
- "컬럼 이름을 알려줘"
"""
                        
                        # 자동으로 기본 데이터 정보 확인 시도
                        if st.session_state.dataframe is not None:
                            try:
                                df = st.session_state.dataframe
                                auto_info = f"""

🔍 **자동 데이터 진단:**
- **데이터 크기**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- **컬럼명**: {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}
- **데이터 타입**: {df.dtypes.value_counts().to_dict()}
- **결측값**: {df.isnull().sum().sum():,} 개
"""
                                user_friendly_error += auto_info
                                    
                            except Exception as info_error:
                                logging.warning(f"Failed to get automatic data info: {info_error}")
                        
                        accumulated_text_obj.append(user_friendly_error)
                        text_placeholder.markdown("".join(accumulated_text_obj))
                        
                        response = {"output": user_friendly_error, "error": error_msg}
                        
                        
                else:
                    # 🔧 langgraph agent는 기존 방식 유지 (이미 비동기) - 03_Agent.py와 동일
                    logging.debug("Processing LangGraph agent query")
                    config = RunnableConfig(
                        recursion_limit=st.session_state.recursion_limit,
                        thread_id=st.session_state.thread_id,
                        configurable={
                            "callbacks": [
                                lambda x: logging.debug(f"RunnableConfig callback: {str(x)[:100]}...")
                            ]
                        }
                    )
                    logging.debug(f"Starting agent execution with timeout: {timeout_seconds}s")
                    
                    # 03_Agent.py와 동일한 방식
                    agent_task = astream_graph(
                        st.session_state.agent,
                        {"messages": [HumanMessage(content=query)]},
                        callback=streaming_callback,
                        config=config,
                    )
                    
                    has_tool_calls = False
                    response = None
                    
                    try:
                        start_time = asyncio.get_event_loop().time()
                        remaining_time = timeout_seconds
                        
                        response = await asyncio.wait_for(
                            agent_task,
                            timeout=remaining_time
                        )
                        
                        logging.debug("Initial agent response received")
                        if "pending_tool_calls" in st.session_state and st.session_state.pending_tool_calls:
                            has_tool_calls = True
                            
                            while st.session_state.pending_tool_calls and remaining_time > 0:
                                logging.debug(f"Processing pending tool calls: {len(st.session_state.pending_tool_calls)}")
                                
                                tool_call = st.session_state.pending_tool_calls[0]
                                logging.debug(f"Processing tool call: {tool_call}")
                                
                                current_time = asyncio.get_event_loop().time()
                                elapsed = current_time - start_time
                                remaining_time = timeout_seconds - elapsed
                                
                                if remaining_time <= 0:
                                    logging.warning("Tool execution timeout")
                                    break
                                
                                # 🔧 수정: MCP 도구들을 그대로 사용 (03_Agent.py 방식)
                                available_tools = []
                                if st.session_state.mcp_client:
                                    # MCP 클라이언트에서 도구 가져오기
                                    mcp_tools = st.session_state.mcp_client.get_tools()
                                    available_tools.extend(mcp_tools)
                                    
                                tool_result = await asyncio.wait_for(
                                    execute_tool(tool_call, available_tools),
                                    timeout=remaining_time
                                )
                                
                                tool_message = ToolMessage(
                                    content=tool_result["content"],
                                    name=tool_result["name"],
                                    tool_call_id=tool_result["tool_call_id"]
                                )
                                
                                with tool_placeholder.expander("🔧 도구 실행 결과", expanded=True):
                                    st.write(f"**도구**: {tool_result['name']}\n\n**결과**:\n```\n{tool_result['content'][:1000]}...\n```")
                                
                                current_time = asyncio.get_event_loop().time()
                                elapsed = current_time - start_time
                                remaining_time = timeout_seconds - elapsed
                                
                                if remaining_time <= 0:
                                    logging.warning("Agent continuation timeout")
                                    break
                                    
                                agent_continue_task = astream_graph(
                                    st.session_state.agent,
                                    {"messages": [tool_message]},
                                    callback=streaming_callback,
                                    config=config,
                                )
                                
                                response = await asyncio.wait_for(
                                    agent_continue_task,
                                    timeout=remaining_time
                                )
                                
                                st.session_state.pending_tool_calls = st.session_state.pending_tool_calls[1:]
                                
                                current_time = asyncio.get_event_loop().time()
                                elapsed = current_time - start_time
                                remaining_time = timeout_seconds - elapsed
                                
                                if not st.session_state.pending_tool_calls:
                                    logging.debug("No more pending tool calls")
                                    break
                                    
                    except asyncio.TimeoutError:
                        error_msg = f"⏱️ 요청 시간이 {timeout_seconds}초를 초과했습니다. 나중에 다시 시도해 주세요."
                        logging.error(f"Query timed out after {timeout_seconds} seconds")
                        return {"error": error_msg}, error_msg, ""
                        
                logging.debug("Query completed successfully")
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
                
            final_text = "".join(accumulated_text_obj)
            final_tool = "".join(accumulated_tool_obj)
            
            # If no streaming content was captured, try to extract from response
            if not final_text and response:
                if isinstance(response, dict):
                    if "output" in response:
                        final_text = str(response["output"])
                    elif "content" in response:
                        final_text = str(response["content"])
                    else:
                        final_text = str(response)
                else:
                    final_text = str(response)
                    
                # Update the placeholder with the final text
                text_placeholder.markdown(final_text)
            
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
    st.session_state.agent_type = None  # 에이전트 타입 (pandas 또는 langgraph)
    st.session_state.history = []  # 대화 기록 저장 리스트
    st.session_state.mcp_client = None  # MCP 클라이언트 객체 저장 공간
    st.session_state.timeout_seconds = 180  # 응답 생성 제한 시간(초), 기본값 120초
    st.session_state.selected_model = "gpt-4o"  # 기본 모델 선택
    st.session_state.recursion_limit = 100  # 재귀 호출 제한, 기본값 100
    st.session_state.selected_prompt_text = ""  # initialize selected prompt text
    st.session_state.temperature = 0.1  # 기본 temperature 설정
    st.session_state.dataframe = None          # 🆕 DataFrame 보관용
    st.session_state.pending_tool_calls = []  # 대기 중인 도구 호출 목록
    st.session_state.tool_responses = {}  # 도구 응답 저장 공간
    st.session_state.current_message_visualizations = []  # 🆕 현재 메시지 시각화 저장

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
                app_found = False
                for section in ai_app_store.get("AIAppStore", []):
                    if app_found:
                        continue
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
                            app_found = True
                            break
                    if app_found:
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