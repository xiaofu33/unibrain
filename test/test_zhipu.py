import os
import sys

# Ensure current dir is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import settings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def main():
    print("========================================")
    print("Testing Zhipu API Connectivity...")
    print(f"API BASE  : {settings.LLM_API_BASE}")
    print(f"API KEY   : {'*' * 10}{settings.LLM_API_KEY[-4:] if len(settings.LLM_API_KEY)>4 else 'INVALID'}")
    print(f"LLM Model : {settings.LLM_MODEL_NAME}")
    print(f"Emb Model : {settings.EMBEDDING_MODEL_NAME}")
    print("========================================")
    
    try:
        # Test Chat API
        print("\n1. 正在测试对话模型大接口 (Chat Completion)...")
        llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            api_key=settings.LLM_API_KEY,  # type: ignore
            base_url=settings.LLM_API_BASE,
            max_tokens=50
        )
        resp = llm.invoke("您好，请用简短的中文回复我：API接通成功。")
        print("✅ 接收到 LLM 回复:")
        print("-----------------")
        print(resp.content)
        print("-----------------")
        
        # Test Embeddings API
        print("\n2. 正在测试文本特征向量化接口 (Embeddings)...")
        emb = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            openai_api_base=settings.LLM_API_BASE,
            openai_api_key=settings.LLM_API_KEY
        )
        vec = emb.embed_query("这是一段用于测试向量化的文本。")
        print(f"✅ 成功生成向量! 返回的隐藏层维度 (Dimension): {len(vec)}")
        print(f"预览前3个浮点数: {vec[:3]}...")
        
        print("\n🎉 === 所有的 API 接口测试均圆满通过 (ALL TESTS PASSED) === 🎉")
    except Exception as e:
        print("\n❌ !!! 接口调用失败 (ERROR DETECTED) !!! ❌")
        print(repr(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
