"""向量存储模块 — ChromaDB + 混元 Embedding"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    SILICONFLOW_EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
)


def get_embeddings() -> OpenAIEmbeddings:
    """返回硅基流动 Embedding 实例（兼容 OpenAI 格式）"""
    return OpenAIEmbeddings(
        model=SILICONFLOW_EMBEDDING_MODEL,
        openai_api_key=SILICONFLOW_API_KEY,
        openai_api_base=SILICONFLOW_BASE_URL,
    )


def get_or_create_vectorstore() -> Chroma:
    """获取或创建 ChromaDB 向量存储"""
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings(),
    )


def add_documents_to_store(docs: list[Document]) -> int:
    """将文档添加到向量库，返回新增的文档 chunk 数量"""
    vectorstore = get_or_create_vectorstore()
    before_count = vectorstore._collection.count()
    vectorstore.add_documents(docs)
    after_count = vectorstore._collection.count()
    return after_count - before_count


def get_document_count() -> int:
    """返回向量库中已存储的文档 chunk 数量"""
    vectorstore = get_or_create_vectorstore()
    return vectorstore._collection.count()


def get_existing_sources() -> set[str]:
    """返回向量库中已入库的文档来源文件名集合"""
    vectorstore = get_or_create_vectorstore()
    if vectorstore._collection.count() == 0:
        return set()
    results = vectorstore._collection.get(include=["metadatas"])
    sources = set()
    for meta in results.get("metadatas", []):
        if meta and "source" in meta:
            sources.add(meta["source"])
    return sources
