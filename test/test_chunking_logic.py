import pytest
from services.document_service import DocumentService
from langchain_core.documents import Document

def test_markdown_chunking_with_context():
    service = DocumentService()
    
    # 模拟一个复杂的 Markdown 文本
    content = """# 第一章：项目背景
这是背景介绍。

## 1.1 业务需求
需求点 1：高性能。
需求点 2：高可用。

### 1.1.1 核心细节
这是一个非常长非常详细的核心细节论述...
""" + "这里是重复性的长文本，用于触发二次分割。" * 100

    doc = Document(page_content=content, metadata={"source": "test.md"})
    
    # 我们模拟 process_file_upload 的内部逻辑或直接调用它（如果能模拟文件系统的化）
    # 这里我们直接手动运行核心逻辑以验证分割效果
    
    # 1. 结构化分割
    header_splits = service.markdown_splitter.split_text(doc.page_content)
    
    final_chunks = []
    for h_split in header_splits:
        headers = [h_split.metadata.get(f"Header {i}") for i in range(1, 4)]
        header_path = " > ".join([h for h in headers if h])
        
        if len(h_split.page_content) > service.text_splitter._chunk_size:
            sub_splits = service.text_splitter.split_text(h_split.page_content)
            for i, sub_content in enumerate(sub_splits):
                context_prefix = f"[章节: {header_path}] " if header_path else ""
                enriched_content = f"{context_prefix}{sub_content}"
                final_chunks.append(Document(page_content=enriched_content, metadata={**h_split.metadata, "chunk_index": i}))
        else:
            context_prefix = f"[章节: {header_path}] " if header_path else ""
            final_chunks.append(Document(page_content=f"{context_prefix}{h_split.page_content}", metadata=h_split.metadata))

    # 验证
    assert len(final_chunks) > 0
    
    # 检查是否包含标题背景
    has_context_1 = any("第一章：项目背景" in c.page_content for c in final_chunks)
    has_context_2 = any("1.1 业务需求" in c.page_content for c in final_chunks)
    has_context_3 = any("1.1.1 核心细节" in c.page_content for c in final_chunks)
    
    assert has_context_1
    assert has_context_2
    assert has_context_3
    
    # 检查长文本子块是否带有前缀
    long_chunk_parts = [c for c in final_chunks if "1.1.1 核心细节" in c.page_content and "章节:" in c.page_content]
    assert len(long_chunk_parts) >= 1
    
    print(f"Total chunks: {len(final_chunks)}")
    for i, c in enumerate(final_chunks[:5]):
        print(f"Chunk {i} Preview: {c.page_content[:100]}...")

if __name__ == "__main__":
    test_markdown_chunking_with_context()
