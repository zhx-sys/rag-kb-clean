"""RAG 私有知识库问答系统 — Gradio 主入口"""

import os
import gradio as gr
from loader import load_documents, split_documents
from vectorstore import add_documents_to_store, get_document_count, get_existing_sources
from chain import build_chain


def ingest_files(files):
    """上传文件并入库"""
    if not files:
        return "请先选择文件。", get_document_count(), None

    try:
        # Gradio 不同版本返回格式可能不同，做兼容处理
        file_paths = []
        for f in files:
            if isinstance(f, str):
                file_paths.append(f)
            elif hasattr(f, "name"):
                file_paths.append(f.name)

        valid_exts = (".pdf", ".docx", ".txt", ".md")
        valid_files = [p for p in file_paths if os.path.splitext(p)[1].lower() in valid_exts]

        if not valid_files:
            return "未找到支持的文档格式（PDF / DOCX / TXT / MD）。", get_document_count(), None

        # 去重 — 过滤已入库的文档
        existing_sources = get_existing_sources()
        new_files = []
        skipped = []
        for f in valid_files:
            fname = os.path.basename(f)
            if fname in existing_sources:
                skipped.append(fname)
            else:
                new_files.append(f)

        if not new_files:
            return (
                f"以下文件已存在，跳过重复入库：\n" + "\n".join(f"  · {n}" for n in skipped)
            ), get_document_count(), None

        docs = load_documents(new_files)
        if not docs:
            return "未能从文件中提取到文本内容。", get_document_count(), None

        chunks = split_documents(docs)
        count = add_documents_to_store(chunks)

        filenames = [os.path.basename(f) for f in new_files]
        msg = f"入库完成：{len(filenames)} 个文件 → {len(chunks)} 个片段（新增 {count} 个 chunk）\n" + \
              "\n".join(f"  · {n}" for n in filenames)
        if skipped:
            msg += f"\n\n跳过重复文件：\n" + "\n".join(f"  · {n}" for n in skipped)
        return msg, get_document_count(), None
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"入库失败：{str(e)}", get_document_count(), None


def chat(message, history, selected_doc):
    """RAG 问答，selected_doc 为选中文档名（全部文档 或 具体文件名）"""
    if not message.strip():
        return ""

    try:
        source = None if selected_doc == "全部文档" else selected_doc
        chain = build_chain(source=source)
        result = chain.invoke(message)

        answer = result["answer"]
        sources = result.get("sources", [])

        # 附加来源引用
        if sources:
            source_lines = []
            seen = set()
            for doc in sources:
                src = doc.metadata.get("source", "未知")
                if src not in seen:
                    seen.add(src)
                    snippet = doc.page_content[:120].replace("\n", " ")
                    source_lines.append(f"> 📄 {src}：{snippet}...")
            if source_lines:
                answer += "\n\n---\n" + "\n".join(source_lines)

        return answer
    except Exception as e:
        return f"问答出错：{str(e)}"


def get_doc_choices():
    """获取下拉框选项列表"""
    sources = get_existing_sources()
    return ["全部文档"] + sorted(sources)


# ---------- Gradio UI ----------
with gr.Blocks(title="RAG 知识库问答") as demo:
    gr.Markdown("# RAG 私有知识库问答系统")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 文档管理")
            file_input = gr.File(
                file_count="multiple",
                label="上传文档（PDF / DOCX / TXT / MD）",
            )
            upload_btn = gr.Button("入库", variant="primary")
            upload_status = gr.Textbox(label="入库状态", interactive=False, lines=5)
            doc_count = gr.Textbox(
                label="知识库 chunk 总数",
                interactive=False,
                value=f"当前：{get_document_count()} 个 chunk",
            )
            doc_dropdown = gr.Dropdown(
                label="选择检索范围",
                choices=get_doc_choices(),
                value="全部文档",
                interactive=True,
            )

            upload_btn.click(
                fn=ingest_files,
                inputs=[file_input],
                outputs=[upload_status, doc_count, file_input],
            ).then(
                fn=lambda: gr.Dropdown(choices=get_doc_choices(), value="全部文档"),
                outputs=[doc_dropdown],
            )

        with gr.Column(scale=2):
            gr.Markdown("### 知识问答")
            chatbot = gr.ChatInterface(
                fn=chat,
                chatbot=gr.Chatbot(height=500),
                textbox=gr.Textbox(
                    placeholder="输入你的问题，基于已入库的知识库回答...",
                    container=False,
                ),
                additional_inputs=[doc_dropdown],
            )

if __name__ == "__main__":
    demo.launch()
