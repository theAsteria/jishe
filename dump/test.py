from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.node_parser.text import (
     SemanticSplitterNodeParser,
     SentenceSplitter,
     TokenTextSplitter,
)
from llama_index.embeddings.ollama import OllamaEmbedding
docs_dir=r"C:\Users\86185\Desktop\计设\mingshi"

# llama_docs = SimpleDirectoryReader(
#         input_dir=docs_dir,
#         # recursive=True,
#         # required_exts=[".txt"] 
# ).load_data()

# load documents
llama_docs = SimpleDirectoryReader(input_files=[r"C:\Users\86185\Desktop\11111.txt"]).load_data()

print(f"加载了 {len(llama_docs)} 个文档")
    
embed_model = OllamaEmbedding(model_name="deepseek-r1:7b",base_url="http://localhost:11434")


splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,  
        buffer_size=1, 
        breakpoint_percentile_threshold=95,
)   
# splitter = SentenceSplitter(
#     chunk_size=512,
#     chunk_overlap=50,
# )
splitter = TokenTextSplitter(
     chunk_size=1024,
     chunk_overlap=20,
     separator=" ",
)

#splitter = LangchainNodeParser(RecursiveCharacterTextSplitter())

# 对每个文档单独进行语义分割
all_nodes = []
for doc in llama_docs:
     # 对单个文档进行分割
     doc_nodes = splitter.get_nodes_from_documents([doc])
     all_nodes.extend(doc_nodes)
     print(f"文档 '{doc.metadata.get('file_name', '未知')}' 分割为 {len(doc_nodes)} 个文本块")

#all_nodes = splitter.get_nodes_from_documents(llama_docs)
print(f"总共创建了 {len(all_nodes)} 个语义文本块")
print(len(all_nodes[0].get_content()))