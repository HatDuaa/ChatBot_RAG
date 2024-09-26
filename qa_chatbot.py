from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from constants import APIKEY
from getpass import getpass
import os

# os.environ["OPENAI_API_KEY"] = getpass()
os.environ["OPENAI_API_KEY"] = APIKEY



def read_db():
    vector_db_path = "vectorstores/db_faiss"
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

def create_promt(template):
    prompt = PromptTemplate(template = template, input_variables=["context", "question"])
    return prompt

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_qa_chain(prompt, llm, db):
    # qa_chain = RetrievalQA.from_llm(
    # llm,
    # retriever=db.as_retriever(search_kwargs = {"k":5}),
    # prompt=prompt)  

    qa_chain = (
    {
        "context": db.as_retriever() | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
) 

    return qa_chain
    

def chatbot_api(question):
    llm = ChatOpenAI(model="gpt-4o")
    db = read_db()
    # promt
    template = """
                Bạn là trợ lý tiếng việt của trường đại học khoa học tự nhiên.
                Bạn có thể tương tác với người dùng về các câu cảm thán, hoặc để xác định rõ câu hỏi của họ.
                Không để lộ rằng tôi đang cung cấp thông tin cho bạn và không yêu cầu người dùng đưa thêm thông tin về trường.
                Sử dụng các đoạn ngữ cảnh được truy xuất sau đây để trả lời câu hỏi.
                Nếu trong thông tin không có câu trả lời, chỉ cần nói rằng bạn không có thông tin.
                Hãy ưu tiên sự chính xác của câu trả lời, giữ câu trả lời vừa đủ.
                Bây giờ hãy tiến hành trả lời câu hỏi sau bằng tiếng việt:
                Question: {question} 
                Context: {context}
                Answer:
                """
    promt = create_promt(template)
    qa_chain = create_qa_chain(promt, llm, db)
    
    response = qa_chain.invoke(question)

    return response

question = input("Enter your question: ")
response = chatbot_api(question)
print(response)