from fastapi import FastAPI, File, UploadFile
from pdf2image import convert_from_bytes
from PIL import Image
import os
import subprocess
from docx2pdf import convert
from langchain_community.document_loaders import PyPDFLoader
app = FastAPI()
target_directory = "D:\\pythonctest\\jianli\\data"
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
os.environ["OPENAI_API_KEY"] = 'sk-***'


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # 创建目标目录
    os.makedirs(target_directory, exist_ok=True)

    # 将上传的文件保存到目标目录
    file_path = os.path.join(target_directory, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # 如果上传的是 PDF 文件，直接将其保存到目标目录
    if file.filename.lower().endswith('.pdf'):
        pdf_path = file_path
    else:
        pdf_filename = f"{os.path.splitext(file.filename)[0]}.pdf"
        pdf_path = os.path.join(target_directory, pdf_filename)
        convert(file_path, pdf_path)
    PDFloader(pdf_path)

    return {"message": "File uploaded and converted to PDF successfully", "pdf_path": pdf_path}


from langchain_community.document_loaders import PyPDFLoader

def PDFloader(path):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    print(pages[0])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
