from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from glob import glob


# download LLaMA2 7B, 13B and 70B for gpu
# wget ./../models/https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf?download=true
# wget ./../models/https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q5_K_M.gguf?download=true
# wget ./../models/https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGUF/resolve/main/llama-2-70b-chat.Q5_K_M.gguf?download=true
 

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
n_gpu_layers = 40 
n_batch = 512


# LLaMA_model = '/mnt/DATA/madara/llama2/llama-2-7b-chat.Q8_0.gguf?download=true'
LLaMA_model = "/mnt/DATA/madara/llama2/llama-2-13b-chat.Q5_K_M.gguf?download=true"
# LLaMA_model = "/mnt/DATA/madara/llama2/llama-2-70b-chat.Q5_K_M.gguf"




def loading_LLM():
    llm = LlamaCpp(
        model_path=LLaMA_model,
        max_tokens=512,
        temperature=0.01,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        top_p=1,
        n_ctx=6000,
        repeat_penalty=1.2,
        callback_manager=callback_manager, 
        verbose=True
    )

    return llm


def load_prompt_for_document():
    template = """Use the provided context to answer the user's question. if you don't know answer then return "I don't know".
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
    return prompt 

def load_prompt_for_dot_metadata():
    template = """If the user ask for any of the above DOT queries, return the generated corresponding answer from the database."".
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
    return prompt 

def load_prompt_for_spel_metadata():
    template = """Use the provided context to answer the user's question. if you don't know answer then return to "I don't know"".
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
    return prompt 


def load_prompt_for_FQL():

    template = """If the user ask for any of the above FQL queries, return the generated corresponding FQL from the database.".
    Context: {context}
    Question: {question}
    Answer:
    """

    prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
    return prompt 


def vector_storage_by_index(db_location):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'}) #TODO: change to GPU
    vdb = FAISS.load_local(db_location, embeddings)
    return vdb


def chain_QA(db_location, promt_pass):
    vdb = vector_storage_by_index(db_location)
    prompt = promt_pass
    retriever = vdb.as_retriever(search_kwargs={'k': 2}) # k is nearest neibhours in vector database search
    chain_return = RetrievalQA.from_chain_type(llm=loading_LLM(),
                                           chain_type='stuff',
                                           retriever=retriever,
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt})
    return chain_return


def get_response(query, chain_res):
    return chain_res({'query': query})['result']

def get_prompt_and_db_location(choice):
    switcher = {
        '1': ('faiss/document', load_prompt_for_document()),
        '2': ('faiss/FQL', load_prompt_for_FQL()),
        '3': ('faiss/dot_metadata', load_prompt_for_dot_metadata()),
        '4': ('faiss/spel_metadata', load_prompt_for_spel_metadata())
    }
    return switcher.get(choice, ('', None))

while True:
    user_input = input('\n\nSelect an option:\n1) Document\n2) Generate FQL\n3) DOT metadata\n4) SPEL metadata\n5) Exit\n\nYour choice: ')

    if user_input == '5' or user_input.lower() == 'exit':
        break



    db_location, promt_pass = get_prompt_and_db_location(user_input)

    if not db_location or not promt_pass:
        print("Invalid choice. Please select a valid option.")
        continue

    chain_qa = chain_QA(db_location, promt_pass)
    user_query = input("\nEnter your query: ")

    current_response = get_response(query=user_query, chain_res=chain_qa)
    print(f'\nAI: {current_response}\n')