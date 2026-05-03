import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "D:\\huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:\\huggingface_cache"

print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")
print(f"HF_HOME: {os.environ.get('HF_HOME')}")

try:
    from sentence_transformers import SentenceTransformer
    
    print("\n正在下载 SciBERT 模型...")
    model = SentenceTransformer(
        'allenai/scibert_scivocab_uncased',
        cache_folder="D:\\huggingface_cache"
    )
    
    print(f"模型加载成功！")
    print(f"向量维度: {model.get_sentence_embedding_dimension()}")
    
    test_embedding = model.encode("This is a test sentence.")
    print(f"测试向量形状: {test_embedding.shape}")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)