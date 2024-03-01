from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader


def start(path):
        
    data_path = "data/" + path
    vdb_path = "faiss/" + path

    pdf_loader = DirectoryLoader(path= data_path , glob="*.pdf", loader_cls=PyPDFLoader)
    dot_loader = DirectoryLoader(path= data_path , glob="*.csv", loader_cls=CSVLoader)
    txt_loader = DirectoryLoader(path= data_path , glob="*.txt", loader_cls=TextLoader)

    pdf_documents = pdf_loader.load()
    dot_documents = dot_loader.load()
    txt_documents = txt_loader.load()
    documents = pdf_documents + dot_documents + txt_documents

    splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                            chunk_overlap=50)
    texts = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)

    db.save_local(vdb_path)
    
db = ['document', 'dot_metadata', 'FQL', 'spel_metadata']

for i in db:
    start(path = i)