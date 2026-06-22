# ================================================================
# RAG 智能问答系统 v11.0 - 各种文本+列表页+详情页完整版
# 支持：列表页自动提取详情链接，逐个抓取详情页内容
# 功能：多路召回 + HyDE + 知识图谱 + Agent + 多模态
# ================================================================

import os
import re
import json
import tempfile
import shutil
import time
import logging
import hashlib
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# ---------- LangChain 核心（兼容所有版本）----------
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader,
    UnstructuredPowerPointLoader, TextLoader,
)
from langchain_community.vectorstores import FAISS

# 兼容 EnsembleRetriever 导入
try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_community.retrievers import EnsembleRetriever
    except ImportError:
        try:
            from langchain.retrievers import EnsembleRetriever
        except ImportError:
            from langchain.retrievers.ensemble import EnsembleRetriever

from langchain_community.retrievers import BM25Retriever

# ---------- 可选扩展 ----------
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

try:
    import easyocr
    import pdfplumber
    _HAS_MULTIMODAL = True
except ImportError:
    _HAS_MULTIMODAL = False

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False

load_dotenv()

# ================================================================
# 日志配置
# ================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("rag.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ================================================================
# 常量
# ================================================================
MAX_FILE_SIZE_MB = 50
REQUEST_TIMEOUT = 30
LLM_TIMEOUT = 60
BASE_INDEX_DIR = Path("knowledge_bases")
BASE_INDEX_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path.home() / ".cache" / "rag_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MAX_HISTORY_TURNS = 5

NOISE_PATTERNS = [
    r'分享\s*.*?微信',
    r'点击\s*.*?关注',
    r'推荐阅读.*?\n',
    r'版权声明.*?\n',
]

PRICE_MAP = {
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "gpt-4": {"input": 30, "output": 60},
    "qwen-plus": {"input": 4, "output": 12},
    "glm-4-flash": {"input": 1, "output": 1},
}
DEFAULT_PRICE = {"input": 1, "output": 2}

# ================================================================
# 工具函数
# ================================================================
def clean_text(text: str) -> str:
    if not text:
        return ""
    for p in NOISE_PATTERNS:
        text = re.sub(p, '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])

def compute_doc_hash(doc: Document) -> str:
    return hashlib.md5(doc.page_content.strip().encode()).hexdigest()[:16]

def estimate_tokens(text: str) -> int:
    cn = len(re.findall(r'[\u4e00-\u9fff]', text))
    en = len(text) - cn
    return int(cn * 1.5 + en * 0.5)

def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}秒"
    return f"{seconds/60:.1f}分钟"

def get_model_price(model: str) -> dict:
    for k in PRICE_MAP:
        if k in model.lower():
            return PRICE_MAP[k]
    return DEFAULT_PRICE

def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    price = get_model_price(model)
    return (input_tokens / 1_000_000) * price["input"] + (output_tokens / 1_000_000) * price["output"]

# ================================================================
# 多模态支持（OCR + 表格提取）
# ================================================================
@st.cache_resource
def get_ocr_reader():
    if _HAS_MULTIMODAL:
        try:
            return easyocr.Reader(['ch_sim', 'en'], gpu=False)
        except:
            pass
    return None

def extract_text_from_image(path: str) -> str:
    reader = get_ocr_reader()
    if reader is None:
        return ""
    try:
        return "\n".join(reader.readtext(path, detail=0))
    except Exception as e:
        logger.warning(f"OCR失败: {e}")
        return ""

def extract_text_from_pdf(path: str) -> str:
    if not _HAS_MULTIMODAL:
        return ""
    try:
        parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                parts.append(page.extract_text() or "")
                for table in page.extract_tables() or []:
                    if table:
                        parts.append(pd.DataFrame(table).to_string())
        return "\n".join(parts)
    except Exception as e:
        logger.warning(f"PDF提取失败: {e}")
        return ""

# ================================================================
# 文档加载（支持图片、PDF增强）
# ================================================================
def load_single_file(file, temp_dir: str) -> List[Document]:
    ext = os.path.splitext(file.name)[1].lower()
    
    # 图片 OCR
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        tmp = os.path.join(temp_dir, file.name)
        with open(tmp, "wb") as f:
            f.write(file.getbuffer())
        text = extract_text_from_image(tmp)
        os.remove(tmp)
        if text:
            return [Document(page_content=clean_text(text), metadata={"source": file.name, "type": "image"})]
        return []
    
    # PDF 增强
    if ext == '.pdf':
        tmp = os.path.join(temp_dir, file.name)
        with open(tmp, "wb") as f:
            f.write(file.getbuffer())
        text = extract_text_from_pdf(tmp)
        if text and len(text) > 100:
            doc = Document(page_content=clean_text(text), metadata={"source": file.name, "type": "pdf"})
            os.remove(tmp)
            return [doc]
        # 回退到 PyPDFLoader
        try:
            loader = PyPDFLoader(tmp)
            docs = loader.load()
            for d in docs:
                d.page_content = clean_text(d.page_content)
                d.metadata["source"] = file.name
            os.remove(tmp)
            return docs
        except:
            os.remove(tmp)
            return []
    
    # 常规文档
    loader_map = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".xls": UnstructuredExcelLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".txt": lambda p: TextLoader(p, encoding='utf-8'),
    }
    if ext not in loader_map:
        return []
    tmp = os.path.join(temp_dir, file.name)
    with open(tmp, "wb") as f:
        f.write(file.getbuffer())
    try:
        loader = loader_map[ext](tmp)
        docs = loader.load()
        for d in docs:
            d.page_content = clean_text(d.page_content)
            d.metadata["source"] = file.name
        return docs
    except Exception as e:
        logger.warning(f"加载失败 {file.name}: {e}")
        return []
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

def load_uploaded_files(uploaded_files, temp_dir: str) -> List[Document]:
    all_docs = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(load_single_file, f, temp_dir) for f in uploaded_files]
        for f in as_completed(futures):
            all_docs.extend(f.result())
    return all_docs

# ================================================================
# 网页加载（支持列表页自动提取详情页）
# ================================================================
def extract_detail_links(soup, base_url) -> List[str]:
    """从列表页提取所有详情页链接（启发式过滤）"""
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        # 构造成绝对 URL
        if href.startswith('/'):
            href = base_url.rstrip('/') + href
        elif not href.startswith('http'):
            continue
        # 过滤：包含常见详情页特征
        if any(pattern in href for pattern in ['/ebooks/', '/book/', '/work/', '/item/', '/detail/', '/read/']):
            links.append(href)
    return list(set(links))  # 去重

def load_webpages(urls_text: str, is_list_page: bool = False, max_detail_pages: int = 50) -> List[Document]:
    """加载网页，支持列表页模式（自动提取并抓取详情页）"""
    urls = [u.strip() for u in urls_text.splitlines() if u.strip() and urlparse(u).scheme]
    all_docs = []
    detail_urls = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for url in urls:
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
            resp.encoding = resp.apparent_encoding or 'utf-8'
            soup = BeautifulSoup(resp.text, "html.parser")

            if is_list_page:
                # 提取详情页链接
                extracted = extract_detail_links(soup, url)
                detail_urls.extend(extracted)
                st.info(f"🔗 从列表页提取到 {len(extracted)} 个详情页链接")
                # 列表页本身不保存为文档（避免噪音）
            else:
                # 普通单页抓取
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                text = clean_text(text)
                if len(text) > 50:
                    all_docs.append(Document(page_content=text[:50000], metadata={"source": url}))
                    st.success(f"✅ 抓取成功: {url[:50]}... ({len(text)} 字符)")
        except Exception as e:
            st.warning(f"⚠️ 加载 {url} 失败: {e}")

    # 如果是列表页模式，抓取所有详情页（限制数量）
    if is_list_page and detail_urls:
        detail_urls = list(set(detail_urls))[:max_detail_pages]
        st.info(f"📚 准备抓取 {len(detail_urls)} 个详情页...")
        progress = st.progress(0)
        for i, detail_url in enumerate(detail_urls):
            try:
                resp = requests.get(detail_url, timeout=REQUEST_TIMEOUT, headers=headers)
                resp.encoding = resp.apparent_encoding or 'utf-8'
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                text = clean_text(text)
                if len(text) > 50:
                    all_docs.append(Document(page_content=text[:50000], metadata={"source": detail_url}))
                    st.success(f"✅ 详情 {i+1}/{len(detail_urls)}: {detail_url[:50]}...")
                else:
                    st.warning(f"⚠️ 详情 {i+1} 内容过短: {detail_url[:50]}...")
                progress.progress((i+1)/len(detail_urls))
                time.sleep(0.3)  # 礼貌延迟
            except Exception as e:
                st.warning(f"⚠️ 详情页 {detail_url} 失败: {e}")
        progress.empty()

    return all_docs

def deduplicate_docs(docs: List[Document]) -> Tuple[List[Document], int]:
    seen, unique = set(), []
    removed = 0
    for d in docs:
        h = compute_doc_hash(d)
        if h not in seen:
            seen.add(h)
            unique.append(d)
        else:
            removed += 1
    return unique, removed

# ================================================================
# 知识图谱（轻量级）
# ================================================================
class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph() if _HAS_NX else None
        self.entity_map = {}
    
    def add_document(self, doc: Document, llm):
        if self.graph is None:
            return
        try:
            prompt = f"从以下文本提取实体和关系，输出JSON列表 [['实体1','关系','实体2'],...]\n文本：{doc.page_content[:1500]}"
            resp = llm.invoke(prompt)
            content = resp.content if hasattr(resp, 'content') else str(resp)
            import re
            match = re.search(r'\[\[.*?\]\]', content, re.DOTALL)
            if match:
                triples = json.loads(match.group())
                for h, r, t in triples:
                    self.graph.add_edge(h, t, relation=r)
                    self.entity_map.setdefault(h, []).append(doc)
                    self.entity_map.setdefault(t, []).append(doc)
        except Exception as e:
            logger.warning(f"KG构建失败: {e}")
    
    def retrieve(self, query: str) -> List[Document]:
        if self.graph is None:
            return []
        entities = re.findall(r'[\u4e00-\u9fa5]{2,}', query)
        if not entities:
            entities = query.split()[:3]
        docs = []
        for e in entities:
            docs.extend(self.entity_map.get(e, []))
        neighbors = set()
        for e in entities:
            if e in self.graph:
                neighbors.update(self.graph.neighbors(e))
                neighbors.update(self.graph.predecessors(e))
        for n in neighbors:
            docs.extend(self.entity_map.get(n, []))
        seen = set()
        unique = []
        for d in docs:
            h = compute_doc_hash(d)
            if h not in seen:
                seen.add(h)
                unique.append(d)
        return unique[:5]

# ================================================================
# HyDE + 多路召回检索器
# ================================================================
class HybridRetriever:
    def __init__(self, vectorstore, llm, embeddings, kg=None, bm25_docs=None, k=10):
        self.vectorstore = vectorstore
        self.llm = llm
        self.embeddings = embeddings
        self.kg = kg
        self.k = k
        self.dense_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k*4}
        )
        if bm25_docs:
            self.bm25_retriever = BM25Retriever.from_documents(bm25_docs)
            self.bm25_retriever.k = k * 2
        else:
            self.bm25_retriever = None
    
    def generate_hypothetical_doc(self, query: str) -> str:
        prompt = f"根据问题生成一段可能包含答案的假设文档（100字左右）：{query}"
        try:
            resp = self.llm.invoke(prompt)
            return resp.content if hasattr(resp, 'content') else str(resp)
        except:
            return query
    
    def invoke(self, query: str) -> List[Document]:
        all_docs = []
        dense = self.dense_retriever.invoke(query)
        for d in dense:
            d.metadata["type"] = "dense"
        all_docs.extend(dense)
        
        if self.bm25_retriever:
            bm = self.bm25_retriever.invoke(query)
            for d in bm:
                d.metadata["type"] = "bm25"
            all_docs.extend(bm)
        
        hypo = self.generate_hypothetical_doc(query)
        hypo_emb = self.embeddings.embed_query(hypo)
        hyde = self.vectorstore.similarity_search_by_vector(hypo_emb, k=self.k)
        for d in hyde:
            d.metadata["type"] = "hyde"
        all_docs.extend(hyde)
        
        if self.kg:
            kg_docs = self.kg.retrieve(query)
            for d in kg_docs:
                d.metadata["type"] = "kg"
            all_docs.extend(kg_docs)
        
        seen = {}
        for d in all_docs:
            key = d.metadata.get("source", "") + d.page_content[:100]
            score = d.metadata.get("relevance_score", 0)
            if key not in seen or score > seen[key].metadata.get("relevance_score", 0):
                seen[key] = d
        result = list(seen.values())
        result.sort(key=lambda x: x.metadata.get("relevance_score", 0), reverse=True)
        return result[:self.k]

# ================================================================
# Agent 工具（简化）
# ================================================================
class SimpleAgent:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    def invoke(self, query: str) -> str:
        if re.search(r'[\d\+\-\*/\(\)]', query) and not re.search(r'[a-zA-Z\u4e00-\u9fa5]', query):
            try:
                result = eval(query, {"__builtins__": {}})
                return f"计算结果: {result}"
            except:
                pass
        if "时间" in query or "现在" in query:
            from datetime import datetime
            return datetime.now().strftime("当前时间: %Y-%m-%d %H:%M:%S")
        docs = self.retriever.invoke(query)
        context = "\n\n".join([d.page_content[:500] for d in docs[:3]])
        prompt = f"""根据上下文回答问题。如果不知道就说不知道。
上下文：{context}
问题：{query}
答案："""
        resp = self.llm.invoke(prompt)
        return resp.content if hasattr(resp, 'content') else str(resp)

# ================================================================
# 主界面
# ================================================================
st.set_page_config(page_title="📚 RAG 智能问答 v11.0", layout="wide")
st.title("📚 文档智能问答系统 v11.0")

for key in ["messages", "chat_history", "vectorstore", "retriever", "doc_count", "llm_config"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["messages", "chat_history"] else None

# ================================================================
# 侧边栏
# ================================================================
with st.sidebar:
    st.header("⚙️ 配置")
    
    platform = st.selectbox("平台", ["阿里云百炼", "智谱AI", "自定义"])
    if "阿里云" in platform:
        default_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        default_model = "qwen-plus"
        default_embed = "text-embedding-v2"
    elif "智谱" in platform:
        default_base = "https://open.bigmodel.cn/api/paas/v4/"
        default_model = "glm-4-flash"
        default_embed = "embedding-3"
    else:
        default_base = "https://api.openai.com/v1/"
        default_model = "gpt-3.5-turbo"
        default_embed = "text-embedding-ada-002"
    
    base_url = st.text_input("Base URL", value=default_base)
    model = st.text_input("模型", value=default_model)
    embed_model = st.text_input("嵌入模型", value=default_embed, 
                               help="阿里云: text-embedding-v2, 智谱: embedding-3, OpenAI: text-embedding-ada-002")
    api_key = st.text_input("API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    
    st.markdown("---")
    st.subheader("📂 知识库")
    
    uploaded_files = st.file_uploader(
        "上传文档 (PDF/Word/Excel/PPT/TXT/图片)",
        accept_multiple_files=True,
        type=["pdf","docx","xlsx","xls","pptx","txt","jpg","jpeg","png","bmp","tiff"]
    )
    urls_text = st.text_area("或输入网页URL（每行一个）", height=80)
    
    # ✅ 新增：列表页模式复选框
    is_list_page = st.checkbox("📋 当前输入为列表页（自动提取所有详情页）", value=False,
                               help="如果输入的是分类页、搜索结果页等包含多个详情链接的页面，勾选此项将自动提取并抓取所有详情页")

    st.subheader("🔍 检索")
    retrieve_k = st.slider("召回数", 1, 20, 5)
    use_hyde = st.checkbox("HyDE (假设文档)", True)
    use_kg = st.checkbox("知识图谱增强", True) if _HAS_NX else st.checkbox("知识图谱增强", False, disabled=True)
    use_mixed = st.checkbox("混合检索 (BM25+向量)", True)
    use_rerank = st.checkbox("重排序", False)
    
    st.subheader("💬 对话")
    enable_history = st.checkbox("多轮记忆", True)
    history_turns = st.slider("记忆轮数", 1, 10, 5, disabled=not enable_history)
    
    col1, col2 = st.columns(2)
    with col1:
        build_btn = st.button("🚀 构建知识库", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ 清空", use_container_width=True)
    
    if clear_btn:
        for k in ["vectorstore", "retriever", "doc_count", "llm_config", "messages", "chat_history"]:
            if k in st.session_state:
                del st.session_state[k]
        st.success("已清空")
        st.rerun()
    
    if st.session_state.vectorstore:
        st.success(f"✅ 已加载，文档块数: {st.session_state.doc_count}")
    else:
        st.info("请构建或加载知识库")

# ================================================================
# 聊天界面
# ================================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================================================================
# 构建知识库
# ================================================================
if build_btn:
    if not api_key:
        st.error("请填写 API Key")
        st.stop()
    if not uploaded_files and not urls_text.strip():
        st.error("请上传文件或输入URL")
        st.stop()
    
    temp_dir = tempfile.mkdtemp()
    try:
        all_docs = []
        if uploaded_files:
            with st.spinner("加载文件..."):
                all_docs.extend(load_uploaded_files(uploaded_files, temp_dir))
        if urls_text.strip():
            with st.spinner("抓取网页..."):
                # ✅ 传入列表页模式参数
                web_docs = load_webpages(urls_text, is_list_page=is_list_page, max_detail_pages=50)
                all_docs.extend(web_docs)
        
        if not all_docs:
            st.error("未加载到有效文档")
            st.stop()
        
        all_docs, removed = deduplicate_docs(all_docs)
        if removed:
            st.info(f"去重移除 {removed} 个")
        
        st.info(f"共 {len(all_docs)} 个文档片段")
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = splitter.split_documents(all_docs)
        split_docs = [d for d in split_docs if d.page_content.strip()]
        st.info(f"分割为 {len(split_docs)} 个块")
        
        embeddings = OpenAIEmbeddings(model=embed_model, api_key=api_key, base_url=base_url)
        texts = [d.page_content for d in split_docs]
        vectors = []
        valid_docs = []
        batch = 32
        for i in range(0, len(texts), batch):
            batch_texts = texts[i:i+batch]
            try:
                emb = embeddings.embed_documents(batch_texts)
                vectors.extend(emb)
                valid_docs.extend(split_docs[i:i+batch])
            except Exception as e:
                st.warning(f"批次嵌入失败: {e}")
                for j, t in enumerate(batch_texts):
                    try:
                        emb = embeddings.embed_query(t)
                        vectors.append(emb)
                        valid_docs.append(split_docs[i+j])
                    except Exception as single_e:
                        st.warning(f"单条嵌入失败: {single_e}")
                        pass
        
        if not vectors:
            st.error("嵌入失败，请检查嵌入模型名称是否正确。")
            st.stop()
        
        vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip([d.page_content for d in valid_docs], vectors)),
            embedding=embeddings,
            metadatas=[d.metadata for d in valid_docs]
        )
        
        kb_dir = BASE_INDEX_DIR / "default"
        kb_dir.mkdir(exist_ok=True)
        vectorstore.save_local(str(kb_dir))
        
        kg = KnowledgeGraph() if use_kg else None
        if kg:
            with st.spinner("构建知识图谱..."):
                llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0)
                for d in all_docs[:20]:
                    kg.add_document(d, llm)
        
        bm25_docs = valid_docs if use_mixed else None
        retriever = HybridRetriever(
            vectorstore=vectorstore,
            llm=ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0),
            embeddings=embeddings,
            kg=kg,
            bm25_docs=bm25_docs,
            k=retrieve_k
        )
        
        st.session_state.vectorstore = vectorstore
        st.session_state.retriever = retriever
        st.session_state.doc_count = len(valid_docs)
        st.session_state.llm_config = {"api_key": api_key, "base_url": base_url, "model": model}
        st.session_state.kg = kg
        
        st.success(f"✅ 构建完成！向量数: {len(valid_docs)}")
        st.rerun()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# ================================================================
# 加载已有知识库
# ================================================================
if not st.session_state.vectorstore:
    kb_dir = BASE_INDEX_DIR / "default"
    if kb_dir.exists() and (kb_dir / "index.faiss").exists():
        if st.button("📂 加载已有知识库"):
            try:
                embeddings = OpenAIEmbeddings(model=embed_model, api_key=api_key, base_url=base_url)
                vectorstore = FAISS.load_local(str(kb_dir), embeddings, allow_dangerous_deserialization=True)
                bm25_path = kb_dir / "bm25_docs.json"
                bm25_docs = None
                if bm25_path.exists():
                    with open(bm25_path, "r") as f:
                        data = json.load(f)
                    bm25_docs = [Document(page_content=d["content"], metadata=d["metadata"]) for d in data]
                retriever = HybridRetriever(
                    vectorstore=vectorstore,
                    llm=ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0),
                    embeddings=embeddings,
                    kg=None,
                    bm25_docs=bm25_docs,
                    k=retrieve_k
                )
                st.session_state.vectorstore = vectorstore
                st.session_state.retriever = retriever
                st.session_state.doc_count = len(vectorstore.index_to_docstore_id) if hasattr(vectorstore, 'index_to_docstore_id') else 0
                st.session_state.llm_config = {"api_key": api_key, "base_url": base_url, "model": model}
                st.success("加载成功")
                st.rerun()
            except Exception as e:
                st.error(f"加载失败: {e}")

# ================================================================
# 问答交互
# ================================================================
if st.session_state.retriever:
    if prompt := st.chat_input("请输入问题"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            placeholder = st.empty()
            start = time.time()
            
            try:
                cfg = st.session_state.llm_config
                llm = ChatOpenAI(
                    model=cfg["model"],
                    api_key=cfg["api_key"],
                    base_url=cfg["base_url"],
                    temperature=0,
                    timeout=LLM_TIMEOUT
                )
                agent = SimpleAgent(llm, st.session_state.retriever)
                answer = agent.invoke(prompt)
                
                elapsed = time.time() - start
                placeholder.markdown(answer)
                
                with st.expander("📄 检索来源"):
                    docs = st.session_state.retriever.invoke(prompt)
                    for i, d in enumerate(docs[:3]):
                        st.markdown(f"**片段 {i+1}** (来源: {d.metadata.get('source','未知')})")
                        st.text(d.page_content[:300] + "...")
                        st.divider()
                
                tokens = estimate_tokens(answer)
                cost = estimate_cost(estimate_tokens(prompt) + estimate_tokens(str(docs)), tokens, cfg["model"])
                st.caption(f"⏱️ {format_time(elapsed)}  |  💰 约 ${cost:.6f}")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                if enable_history:
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    if len(st.session_state.chat_history) > history_turns * 2:
                        st.session_state.chat_history = st.session_state.chat_history[-history_turns*2:]
                
            except Exception as e:
                st.error(f"错误: {e}")

# ================================================================
# 监控仪表盘
# ================================================================
with st.sidebar:
    st.markdown("---")
    if st.session_state.messages:
        total_qa = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.caption(f"📊 本次会话问答数: {total_qa}")
    st.caption(f"v11.0  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")