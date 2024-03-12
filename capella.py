import streamlit as st
import os
import openai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from azure.search.documents.models import VectorizableTextQuery
from azure.search.documents.models import RawVectorQuery
from azure.search.documents.models import RawVectorQuery
from azure.core.credentials import AzureKeyCredential

############################################################## Log-IN Module##########################################################
## Function to check login credentials
# def authenticate(password):
#    # Replace with your authentication logic
#    # For simplicity, a hardcoded email and password are used
#    #valid_email = "ibchelpdesk@ibc.com"
#    valid_password = "ibcsampleapp@123"
    
#    return password == valid_password
# # Session state initialization
# if 'logged_in' not in st.session_state:
#    st.session_state.logged_in = False
# # Login Page
# login = st.sidebar.checkbox("Login")
# if login and not st.session_state.logged_in:
#    st.sidebar.title("Login")
#    #email = st.sidebar.text_input("Email")
#    password = st.sidebar.text_input("Password", type="password")
#    if st.sidebar.button("Login"):
#        if authenticate(password):
#            st.session_state.logged_in = True
#            st.experimental_rerun()
#        else:
#            st.sidebar.error("Invalid password")
# # Check if the user is logged in before proceeding
# if not st.session_state.logged_in:
#    st.warning("Please log in to use the IBC Assistant.")
#    st.stop()  # Stop further execution if not logged in

############################################################## Log-IN Module ##########################################################    

st.title("Student Financial Aid Assistant")




OPENAI_API_KEY = "1023355b5d1845a6a83163b02be2fd3f"
OPENAI_API_ENDPOINT = "https://newopenaineom.openai.azure.com/"
# OPENAI_API_VERSION = "2023-09-01-preview"
OPENAI_API_VERSION = "2024-02-15-preview"

AZURE_COGNITIVE_SEARCH_SERVICE_NAME = "saudicog "
AZURE_COGNITIVE_SEARCH_API_KEY = "WcoHsHqc67FfZrK56XDUBgpuii2nU6zhAOBBIfDU2fAzSeCnDSXE"
AZURE_COGNITIVE_SEARCH_ENDPOINT = "https://saudicog.search.windows.net"
azure_credential = AzureKeyCredential(AZURE_COGNITIVE_SEARCH_API_KEY)

AZURE_COGNITIVE_SEARCH_INDEX_NAME = "capella_financial_aid"
#AZURE_COGNITIVE_SEARCH_INDEX_NAME = "process_all_category_index"



#logo_url = "https://www.ibc.com/images/ibc-logo.png"
#logo_url = "https://www.google.com/url?sa=i&url=https%3A%2F%2Fpoonawallafincorp.com%2Fblogs%2Fbusiness-finances-meaning-sources-and-types.php&psig=AOvVaw0yhNHLXsIU9Ppf8dbkGZGJ&ust=1709196928553000&source=images&cd=vfe&opi=89978449&ved=0CBMQjRxqFwoTCKCGk4PVzYQDFQAAAAAdAAAAABAE"
# logo_url = "https://www.capella.edu/"

# logo_html = f'<img src="{logo_url}" alt="Logo" height="130" width="250">'
# st.sidebar.markdown(f'<div class="logo-container">{logo_html}</div>', unsafe_allow_html=True)

logo_url = "https://info.elitecme.com/hubfs/image-png-1.png"
logo_html = f'<img src="{logo_url}" alt="Logo" height="200" width="300">'
st.sidebar.markdown(f'<div class="logo-container">{logo_html}</div>', unsafe_allow_html=True)



dropdown_3_prompt = " "




# category = st.sidebar.selectbox('Select Index',("Invoice", "POs" ,"Receipt"))

# if category == "Invoice":
#     AZURE_COGNITIVE_SEARCH_INDEX_NAME  = "invoice"
    
#     #dropdown_3_prompt = " "

# if category == "POs":
#     AZURE_COGNITIVE_SEARCH_INDEX_NAME = "po"
    
    
# if category == "Receipt":
#     AZURE_COGNITIVE_SEARCH_INDEX_NAME = "receipt"


use_memory = st.sidebar.checkbox('Enable Memory')

    #st.session_state.messages = []
if st.sidebar.button(':red[New Topic]'):
    st.session_state.messages = []


with st.sidebar:
    st.write("""
    **Recommended Prompts**-
             
    * How is the monthly payment amount calculated under the ICR Plan?
    * What are the types of income-driven repayment plans offered?
    * Am i eligible for student aid if my permanent residence status of US has expired?

    """)
with st.expander("See Walkthrough"):
    st.write("""
        Video
    """)
    st.video("Student_financial_aid_assistant.mp4")

    
    
    
######################################################### Neom ##########################################################    

######################################################### Neom ##########################################################    
    

######################################################### Neom ##########################################################

######################################################### Neom ##########################################################
import os
from openai import AzureOpenAI

client = AzureOpenAI(
  api_key = OPENAI_API_KEY,  
  api_version = OPENAI_API_VERSION,
  azure_endpoint = OPENAI_API_ENDPOINT
)

def generate_embeddings_azure_openai(text = " "):
    response = client.embeddings.create(
        input = text,
        model= "ada2new"
    )
    return response.data[0].embedding




def call_gpt_model(model= "gpt35turbo",
                                  messages= [],
                                  temperature=0.1,
                                  max_tokens = 700,
                                  stream = True):

    print("Using model :","gpt35turbo")

    response = client.chat.completions.create(model="gpt35turbo",
                                              messages=messages,
                                              temperature = temperature,
                                              max_tokens = max_tokens,
                                              stream= stream)

    return response
    
system_message_query_generation_for_retriver = """
You are a very good text analyzer.
You will be provided a chat history and a user question.
You task is generate a search query that will return the best answer from the knowledge base.
Try and generate a grammatical sentence for the search query.
Do NOT use quotes and avoid other search operators.
Do not include cited source filenames and document names such as info.txt or doc.pdf in the search query terms.
Do not include any text inside [] or <<>> in the search query terms.
"""


def generate_query_for_retriver(user_query = " ",messages = []):

    start = time.time()
    user_message = summary_prompt_template = """Chat History:
    {chat_history}

    Question:
    {question}

    Search query:"""

    user_message = user_message.format(chat_history=str(messages), question=user_query)

    chat_conversations_for_query_generation_for_retriver = [{"role" : "system", "content" : system_message_query_generation_for_retriver}]
    chat_conversations_for_query_generation_for_retriver.append({"role": "user", "content": user_message })

    response = call_gpt_model(messages = chat_conversations_for_query_generation_for_retriver,stream = False ).choices[0].message.content
    print("Generated Query for Retriver in :", time.time()-start,'seconds.')
    print("Generated Query for Retriver is :",response)

    return response
    
    
class retrive_similiar_docs : 

    def __init__(self,query = " ", retrive_fields = ["actual_content", "metadata"],
                      ):
        if query:
            self.query = query

        self.search_client = SearchClient(AZURE_COGNITIVE_SEARCH_ENDPOINT, AZURE_COGNITIVE_SEARCH_INDEX_NAME, azure_credential)
        self.retrive_fields = retrive_fields
    
    def text_search(self,top = 2):
        results = self.search_client.search(search_text= self.query,
                                select=self.retrive_fields,top=top)
        
        return results
        

    def pure_vector_search(self, k = 2, vector_field = 'vector',query_embedding = []):

        vector_query = RawVectorQuery(vector=query_embedding, k=k, fields=vector_field)

        results = self.search_client.search( search_text=None,  vector_queries= [vector_query],
                                            select=self.retrive_fields)

        return results
        
    def hybrid_search(self,top = 2, k = 2,vector_field = "vector",query_embedding = []):
        
        vector_query = RawVectorQuery(vector=query_embedding, k=k, fields=vector_field)
        results = self.search_client.search(search_text=self.query,  vector_queries= [vector_query],
                                                select=self.retrive_fields,top=top)  

        return results



import time
start = time.time()


def get_similiar_content(user_query = " ",
                      search_type = "hybrid",top = 2, k =2):

    #print("Generating query for embedding...")
    #embedding_query = get_query_for_embedding(user_query=user_query)
    retrive_docs = retrive_similiar_docs(query = user_query)

    if search_type == "text":
        start = time.time()
        r = retrive_docs.text_search(top =top)

        sources = []
        similiar_doc = []
        for doc in r:
            similiar_doc.append(doc["metadata"] + ": " + doc["actual_content"].replace("\n", "").replace("\r", ""))
            sources.append(doc["metadata"])
        similiar_docs = "\n".join(similiar_doc)
        print("Retrived similiar documents with text search in :", time.time()-start,'seconds.')
        #print("Retrived Docs are :",sources,"\n")

    if search_type == "vector":
        start = time.time()
        vector_of_search_query = generate_embeddings_azure_openai(user_query)
        print("Generated embedding for search query in :", time.time()-start,'seconds.')

        start = time.time()
        r = retrive_docs.pure_vector_search(k=k, query_embedding = vector_of_search_query)

        sources = []
        similiar_doc = []
        for doc in r:
            similiar_doc.append(doc["metadata"] + ": " + doc["actual_content"].replace("\n", "").replace("\r", ""))
            sources.append(doc["metadata"])
        similiar_docs = "\n".join(similiar_doc)
        print("Retrived similiar documents with text search in :", time.time()-start,'seconds.')
       # print("Retrived Docs are :",sources,"\n")


    if search_type == "hybrid":
        start = time.time()
        vector_of_search_query = generate_embeddings_azure_openai(user_query)
        print("Generated embedding for search query in :", time.time()-start,'seconds.')

        start = time.time()
        r = retrive_docs.hybrid_search(top = top, k=k, query_embedding = vector_of_search_query)

        sources = []
        similiar_doc = []
        for doc in r:
            similiar_doc.append(doc["metadata"] + ": " + doc["actual_content"].replace("\n", "").replace("\r", ""))
            sources.append(doc["metadata"])
        similiar_docs = "\n".join(similiar_doc)
        print("*"*100)
        print("Retrived similiar documents with text search in :", time.time()-start,'seconds.')
        #print("similiar_doc :", similiar_doc)
        #print("Retrived Docs are :",sources,"\n")
        #print("similiar_doc :", similiar_doc)
        #print("*"*100)
    return similiar_docs
    

def stream_response(stream_object):
    full_response = " "
    for chunk in stream_object:
        if len(chunk.choices) >0:
            if str(chunk.choices[0].delta.content) != "None": 
                full_response += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content,end = '')
    return full_response


def generate_response_without_memory(user_query = " ",stream = True,max_tokens = 512,model = " "):

    similiar_docs = get_similiar_content(user_query = user_query)
    user_content = user_query + " \nSOURCES:\n" + similiar_docs
    chat_conversations = [{"role" : "system", "content" : system_message}]
    chat_conversations.append({"role": "user", "content": user_content })
    response = call_gpt_model(messages = chat_conversations,stream = stream,max_tokens=max_tokens)
    #response = stream_response(response)
    return response


system_message = """
Assistant provides precise answers in points based on capella university's financial aid documents. Be concise in your responses, sticking strictly to the facts from the listed sources below. If information is insufficient, indicate that you don't know.""" + dropdown_3_prompt + """

Please always answer exactly what the user asks, and avoid unnecessary details. Reference sources by including at least 2 new line characters followed by the source in square brackets, like this: "\n\n [ Source : info.txt]". 

Give proper responses to greetings like .. Hello, How are you? etc.
"""

chat_conversations_global_message = [{"role" : "system", "content" : system_message}]


def generate_response_with_memory(user_query = " ",keep_messages = 10,new_conversation = False,model="ibcgpt35turbo",stream=False):

    #global chat_conversations_to_send

#    if new_conversation:
#        st.session_state.messages = []

    
    #print(CHAT_CONVERSATION_TO_SEND)
    #print(chat_conversations_to_send)

    query_for_retriver = generate_query_for_retriver(user_query=user_query,messages = st.session_state.messages[-keep_messages:])
    similiar_docs = get_similiar_content(query_for_retriver)
    #print("Query for Retriver :",query_for_retriver)
    similiar_docs = get_similiar_content(query_for_retriver)
    user_content = user_query + " \nSOURCES:\n" + similiar_docs

    chat_conversations_to_send = chat_conversations_global_message + st.session_state.messages[-keep_messages:] + [{"role":"user","content" : user_content}]
    
    response_from_model = call_gpt_model(messages = chat_conversations_to_send)
    #print("Response_from_model :",response_from_model)
    #chat_conversations_to_send = chat_conversations_to_send[1:]
    #st.session_state.messages[-1] = {"role": "user", "content": user_query}
    #st.session_state.messages.append({"role": "assistant", "content": response_from_model})
    #print("*"*100)
    #print("chat_conversations_to_send :", chat_conversations_to_send)
    #print("*"*100)

    return response_from_model


    

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
else:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            avatar = "🤖"
        else:
            avatar = "🧑‍💻"
        with st.chat_message(message["role"],avatar = avatar ):
            st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask me anything..."):
    # Display user message in chat message container
    st.chat_message("user",avatar = "🧑‍💻").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    user_query = prompt + dropdown_3_prompt
    if use_memory:
        response = generate_response_with_memory(user_query= user_query,stream=True)
    else :
        response = generate_response_without_memory(user_query= user_query,stream=True)

    with st.chat_message("assistant",avatar = "🤖"):
        message_placeholder = st.empty()
        full_response = " "
        # Simulate stream of response with milliseconds delay
        for chunk in response:
            if len(chunk.choices) >0:
                if str(chunk.choices[0].delta.content) != "None": 
                    full_response += chunk.choices[0].delta.content
                    #message_placeholder.markdown(full_response + "▌")
        full_response = full_response.replace("$", "\$")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
