# export LANGCHAIN_TRACING_V2="true"
# export LANGCHAIN_API_KEY=“ls__cd1b2ed501394e30aecacdbafb38bc2d”

from langchain.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import WebBaseLoader, PagedPDFSplitter, UnstructuredWordDocumentLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain, ConversationalRetrievalChain

DOC_PATH = "../documents/"
vec_res = [
    {
        'type': 'url',
        'path': 'https://www.canadim.com/immigrate/provincial-nominee-program/ontario/ontario-express-entry/'
                'human-capital-priorities/'
    }, {
        'type': 'url',
        'path': 'https://www.ontario.ca/page/ontarios-express-entry-human-capital-priorities-stream'
    }, {
        'type': 'pdf',
        'path': DOC_PATH + 'Immigration and Refugee Protection Act.pdf'
    }, {
        'type': 'pdf',
        'path': DOC_PATH + 'Immigration and Refugee Protection Regulations.pdf'
    }, {
        'type': 'pdf',
        'path': DOC_PATH + 'Immigration Appeal Division Rules.pdf'
    }, {
        'type': 'pdf',
        'path': DOC_PATH + 'Immigration Division Rules.pdf'
    }, {
        'type': 'pdf',
        'path': DOC_PATH + 'Oath or Solemn Affirmation of Office Rules.pdf'
    }, {
        'type': 'pdf',
        'path': DOC_PATH + 'Refugee Appeal Division Rules.pdf'
    }, {
        'type': 'pdf',
        'path': DOC_PATH + 'Refugee Protection Division Rules.pdf'
    }, {
        'type': 'pdf',
        'path': DOC_PATH + 'Regulatory Impact Analysis Statement - Immigration and Refugee Board of Canada.pdf'
    }, {
        'type': 'word',
        'path': DOC_PATH + 'ONTARIO REGULATION 421:17.doc'
    }, {
        'type': 'word',
        'path': DOC_PATH + 'ONTARIO REGULATION 422:17.doc'
    }, {
        'type': 'word',
        'path': DOC_PATH + 'Ontario Immigration Act.doc'
    }
]


def get_webpage(url):
    # todo... adopt distributed web crawler here
    loader = WebBaseLoader(url)
    return loader.load()


def get_pdf(path):
    loader = PagedPDFSplitter(path)
    return loader.load()


def get_word(path):
    loader = UnstructuredWordDocumentLoader(path)
    return loader.load()


def gen_vectorstore():
    docs = []
    for doc in vec_res:
        dtype = doc['type']
        if dtype == 'url':
            docs.extend(get_webpage(doc['path']))
        elif dtype == 'pdf':
            docs.extend(get_pdf(doc['path']))
        elif dtype == 'word':
            docs.extend(get_word(doc['path']))

    embeddings = OllamaEmbeddings()

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    return FAISS.from_documents(documents, embeddings)


if __name__ == '__main__':
    vector = gen_vectorstore()

    # q = 'how can I get PR through Human Capital Priorities stream in Ontario, Canada?'
    # q = 'Which NOC codes are approved for the Human Capital Priorities stream in Ontario, targeted to tech workers?'
    # q = '这是个加拿大移民相关的话题。最大的政策洼地不是法语吗，法语政策的倾斜也会改变吗？'
    q = ('What\'s the criteria for an applicant for a certificate of nomination in the master’s '
         'graduate category in Ontario?')

    llm = Ollama(model='llama2')
    print('', 'step1', llm.invoke(q))

    # guide llm's response with a prompt template to convert raw user input to better input to the LLM.
    # prompt = ChatPromptTemplate.from_messages([
    #     ('system', 'You are an immigration lawyer in Canada.'),
    #     ('user', '{input}')
    # ])
    #
    # chain = prompt | llm
    # print('', 'step2', chain.invoke({'input': q}))
    #
    # output_parser = StrOutputParser()
    #
    # chain = chain | output_parser
    # print('', 'step3', chain.invoke({'input': q}))

    prompt = ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context:
            <context>
            {context}
            </context>
        Question: {input}"""
    )

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    # todo... to add a conversational retrieval chain
    # conversational_retrieval_chain = ConversationalRetrievalChain()
    response = retrieval_chain.invoke({'input': q})
    print(response['answer'])
