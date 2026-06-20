"""RAG 问答链 — LCEL 管道"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    TOP_K,
)
from vectorstore import get_or_create_vectorstore


SYSTEM_PROMPT = (
    '你是一个基于私有知识库的问答助手。\n'
    '请根据以下参考资料回答用户问题。如果参考资料中没有相关信息，'
    '请如实告知"知识库中未找到相关内容"，不要编造。\n\n'
    '参考资料：\n{context}'
)


def get_llm() -> ChatOpenAI:
    """返回 DeepSeek LLM 实例"""
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_BASE_URL,
        temperature=0.3,
    )


def format_docs(docs):
    """将文档列表拼接为上下文字符串"""
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(source: str = None):
    """构建 RAG 链，返回 answer 和 sources。
    
    source: 指定文档名过滤，None 或空字符串表示全部文档
    """
    vectorstore = get_or_create_vectorstore()
    
    search_kwargs = {"k": TOP_K}
    if source:
        search_kwargs["filter"] = {"source": source}
    
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{input}")]
    )

    # LCEL 管道：检索 → 拼接上下文 → LLM 生成
    rag_chain = (
        {
            "context": retriever,
            "input": RunnablePassthrough(),
        }
        | RunnableParallel(
            answer=(
                RunnablePassthrough.assign(
                    context=lambda x: format_docs(x["context"])
                )
                | prompt
                | llm
                | StrOutputParser()
            ),
            sources=lambda x: x["context"],
        )
    )

    return rag_chain
