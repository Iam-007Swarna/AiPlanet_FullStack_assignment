import streamlit as st
import random
import os
import time
# from google.cloud import storage
from Astronomy_BH_hybrid_RAG import get_query

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "tidy-resolver-411707-0f032726c297.json"


# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     """Downloads a blob from the bucket."""
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#     blob.download_to_filename(destination_file_name)
#     print(f"Downloaded storage object {source_blob_name} from bucket {bucket_name} to local file {destination_file_name}.")

# if not (os.path.exists("storage_file/bm25") and os.path.exists("storage_file/kg")):
#     # List of file names to download
#     file_names = [
#         "default__vector_store.json",
#         "docstore.json",
#         "graph_store.json",
#         "image__vector_store.json",
#         "index_store.json"
#     ]

#     # Bucket name
#     bucket_name = "title_tailors_bucket"

#     # Create the destination directory if it doesn't exist
#     os.makedirs("storage_file/bm25", exist_ok=True)

#     # Loop through the file names and download each one
#     for file_name in file_names:
#         source_blob_name = f"storage/bm25/{file_name}"
#         destination_file_name = f"storage_file/bm25/{file_name}"
#         download_blob(bucket_name, source_blob_name, destination_file_name)
        
#     # List of file names to download
#     file_names = [
#         "default__vector_store.json",
#         "docstore.json",
#         "graph_store.json",
#         "image__vector_store.json",
#         "index_store.json"
#     ]

#     # Bucket name
#     bucket_name = "title_tailors_bucket"

#     # Create the destination directory if it doesn't exist
#     os.makedirs("storage_file/kg", exist_ok=True)

#     # Loop through the file names and download each one
#     for file_name in file_names:
#         source_blob_name = f"storage/kg/{file_name}"
#         destination_file_name = f"storage_file/kg/{file_name}"
#         download_blob(bucket_name, source_blob_name, destination_file_name)
# else:
#     print("Files already exist in the storage_file directory.")

# Streamed response emulator
def response_generator(text):
    output = get_query(text)
    responses = {
        "Knowledge Graph Response" : output[1],
        "Dense + BM25 without KG Response" : output[2],
        "Dense + BM25 with KG Response" : output[2]
    }
    return responses

st.title("Context-aware Astronomy ChatBot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    response_dict = response_generator(prompt)

    # Display each part of the response in separate markdown blocks with headings
    with st.chat_message("assistant"):
        for key, value in response_dict.items():
            st.markdown(f"### {key}")
            st.markdown(value)

    # Add assistant response to chat history
    for key, value in response_dict.items():
        st.session_state.messages.append({"role": "assistant", "content": f"### {key}\n{value}"})
