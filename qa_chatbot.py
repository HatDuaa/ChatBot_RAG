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
    prompt = PromptTemplate(template = template, input_variables=["context", "question"])
    return prompt

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


### Đặt câu hỏi theo ngữ cảnh ###
def create_history(llm, retriever):
    contextualize_q_system_prompt = r"""Đưa ra lịch sử trò chuyện và câu hỏi mới nhất của người dùng \ 
    có thể tham chiếu ngữ cảnh trong lịch sử trò chuyện, hãy tạo một câu hỏi độc lập \ 
    cái mà có thể hiểu được nếu không có lịch sử trò chuyện. KHÔNG trả lời câu hỏi, \ 
    chỉ định dạng lại câu hỏi nếu cần và nếu không thì trả lại nguyên trạng"""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

   
    return history_aware_retriever

def create_rag_chain(history_aware_retriever, llm):
    qa_system_prompt = """
                    Bạn là trợ lý của trường đại học khoa học tự nhiên.
                    Bạn có thể tương tác với người dùng về các câu cảm thán, hoặc để xác định rõ câu hỏi của họ.
                    Không để lộ rằng tôi đang cung cấp thông tin cho bạn và không yêu cầu người dùng đưa thêm thông tin về trường.
                    Sử dụng các đoạn ngữ cảnh được truy xuất sau đây để trả lời câu hỏi.
                    Nếu trong thông tin không có câu trả lời, chỉ cần nói rằng bạn không có thông tin.
                    Hãy ưu tiên sự chính xác của câu trả lời, giữ câu trả lời vừa đủ.

                    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain
    

### Quản lí lích sử chat ###
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id] 

### Tạo chatbot ###
def chatbot_api(question, session_id):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    db = read_db()
    retriever = db.as_retriever()
    
    history_aware_retriever = create_history(llm, retriever)
    rag_chain = create_rag_chain(history_aware_retriever, llm)
    
    conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
    )

    answer = conversational_rag_chain.invoke(
                {"input": question},
                config={
                    "configurable": {"session_id": session_id}
                },  # constructs a key in `store`.
            )["answer"]

    return answer

question = ""
while question != "end":
    question = input("Enter your question: ")
    if question != "end":
        response = chatbot_api(question, "abc123")
        print(response)
        #print(store)