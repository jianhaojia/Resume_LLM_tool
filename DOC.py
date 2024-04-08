from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
import logging

# 创建日志记录器
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # 设置日志记录器的级别为 DEBUG

# 创建控制台处理程序
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)  # 设置控制台处理程序的日志级别为 ERROR

# 创建日志格式器
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# 将控制台处理程序添加到日志记录器
logger.addHandler(console_handler)
os.environ["OPENAI_API_KEY"] = 'sk-***'
# 导入文本
loader = PyPDFLoader("D:\\pythonctest\\jianli\\data\\157.pdf")
# 将文本转成 Document 对象
document = loader.load()
# 初始化加载器
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# 切割加载的 document
split_docs = text_splitter.split_documents(document)

# 初始化 openai 的 embeddings 对象
embeddings = OpenAIEmbeddings()
# 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
docsearch = Chroma.from_documents(split_docs, embeddings)

# 创建问答对象
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)
# 进行问答
result = qa({"query": "获取下面这段信息里面关于名字的信息并返回出来，直接返回名字，多余信息不需要，格式为：答案"})
print(result['result'])
result = qa({"query": "获取下面这段信息里面关于期望岗位的信息并返回出来，并返回他的期望岗位，只返回该岗位，不要返回多余的信息，格式为：答案"})
print(result['result'])