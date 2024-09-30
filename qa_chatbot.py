from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

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
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )
    return prompt

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


### Đặt câu hỏi theo ngữ cảnh ###
def create_history(llm, retriever):
    contextualize_q_system_prompt = r"""Đưa ra lịch sử trò chuyện và câu hỏi mới nhất của người dùng \ 
    có thể tham chiếu ngữ cảnh trong lịch sử trò chuyện, hãy tạo một câu hỏi độc lập \ 
    cái mà có thể hiểu được nếu không có lịch sử trò chuyện. KHÔNG trả lời câu hỏi, \ 
    chỉ định dạng lại câu hỏi nếu cần và nếu không thì trả lại nguyên trạng"""
    
    contextualize_q_prompt = create_promt(contextualize_q_system_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    return history_aware_retriever

def create_rag_chain(history_aware_retriever, llm):
    qa_system_prompt = """
                    Bạn là trợ lý trả lời câu hỏi của trường Đại Học Khoa Học Tự Nhiên - Đại Học Quốc Gia Thành Phố Hồ Chí Minh.
                    Bạn có thể tương tác với người dùng về các câu cảm thán, hoặc để xác định rõ câu hỏi của họ.
                    Không để lộ rằng tôi đang cung cấp thông tin cho bạn và không yêu cầu người dùng đưa thêm thông tin về trường.
                    Sử dụng các đoạn ngữ cảnh được truy xuất sau đây để trả lời câu hỏi.
                    Nếu trong thông tin không có câu trả lời, chỉ cần nói rằng bạn không có thông tin.
                    Hãy ưu tiên sự chính xác của câu trả lời, giữ câu trả lời vừa đủ.

                    {context}"""
    
    
    qa_prompt = create_promt(qa_system_prompt)

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
    )

    return conversational_rag_chain




### Quản lí lích sử chat ###
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def delete_session_history(session_id: str):
    if session_id in store:
        del store[session_id]

### Tạo chatbot ###
def chatbot_api(question, session_id):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    db = read_db()
    retriever = db.as_retriever(top_k=3)
    
    history_aware_retriever = create_history(llm, retriever)
    conversational_rag_chain = create_rag_chain(history_aware_retriever, llm)

    generator = conversational_rag_chain.stream(
                {"input": question},
                config={
                    "configurable": {"session_id": session_id}
                },  # constructs a key in `store`.
            )
    
    answer = ""

    for chunk in generator:
        if "answer" in chunk:
            answer += chunk["answer"]
            print(chunk["answer"], end="", flush=True)  # In ra từng phần của câu trả lời
    print()


    return answer




### Chạy chatbot ###
question = ""
while question != "end":
    session_id = input("Enter your session id: ")
    question = input("Enter your question: ")
    if question != "end":
        response = chatbot_api(question, session_id)
        #print(response)
        #print(store)