# export LANGCHAIN_TRACING_V2="true"
# export LANGCHAIN_API_KEY=“ls__cd1b2ed501394e30aecacdbafb38bc2d”

from langchain.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain


def get_webpage():
    # https://www.ontario.ca/page/ontarios-express-entry-human-capital-priorities-stream
    loader = WebBaseLoader("https://www.canadim.com/immigrate/provincial-nominee-program/ontario/ontario-express-entry/human-capital-priorities/")
    docs = loader.load()
    print(docs)
    return docs


def gen_vectorstore():
    docs = get_webpage()

    embeddings = OllamaEmbeddings()

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    return FAISS.from_documents(documents, embeddings)


if __name__ == '__main__':
    vector = gen_vectorstore()

    # q1 = "how can I get PR through Human Capital Priorities stream in Ontario, Canada?"
    # q1 = "Which NOC codes are approved for the Human Capital Priorities stream in Ontario, targeted to tech workers?"
    q1 = "这是个加拿大移民相关的话题。最大的政策洼地不是法语吗，法语政策的倾斜也会改变吗？"

    llm = Ollama(model="llama2")
    print('', 'step1', llm.invoke(q1))

    # guide llm's response with a prompt template to convert raw user input to better input to the LLM.
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", "You are an immigration lawyer in Canada."),
    #     ("user", "{input}")
    # ])
    #
    # chain = prompt | llm
    # print('', 'step2', chain.invoke({"input": q1}))
    #
    # output_parser = StrOutputParser()
    #
    # chain = chain | output_parser
    # print('', 'step3', chain.invoke({"input": q1}))

    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": q1})
    print(response["answer"])
