"""文档加载与分割模块"""

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(file_paths: list[str]) -> list[Document]:
    """根据文件扩展名选择对应的加载器，返回文档列表"""
    docs = []
    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        loader = None

        if ext == ".pdf":
            loader = PyPDFLoader(path)
        elif ext == ".docx":
            loader = Docx2txtLoader(path)
        elif ext in (".txt", ".md", ".markdown"):
            loader = TextLoader(path, encoding="utf-8")
        else:
            print(f"  跳过不支持的文件类型: {path}")
            continue

        loaded = loader.load()
        # 为每个文档片段标注来源文件名
        filename = os.path.basename(path)
        for doc in loaded:
            doc.metadata["source"] = filename
        docs.extend(loaded)

    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """分割文档为 chunk"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    return splitter.split_documents(docs)
