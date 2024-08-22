import requests
import json
import streamlit as st

import json
import time
import uuid
import hashlib
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import VectorParams, Distance
from openai import OpenAI

# Constants
QDRANT_URL = 'https://qdrant.utvecklingfalkenberg.se'
QDRANT_PORT = 443
EMBEDDING_MODEL = "text-embedding-3-large"  # Using the larger model
BATCH_SIZE = 1000
SLEEP_TIME = 1
VECTOR_SIZE = 3072  # Updated vector size for large embeddings
COLLECTION_NAME = 'mediawiki'  # Collection name for MediaWiki data
# Constants MediaWiki
USERNAME = "Api01@api"
PASSWORD = st.secrets["MEDIAWIKI_API_KEY"]
URL = "https://wiki01.utvecklingfalkenberg.se/api.php"

# Constants qdrant / openai
qdrant_api_key = st.secrets['QDRANT_API_KEY']
openai_api_key = st.secrets['OPENAI_API_KEY']

qdrant_client = QdrantClient(url=QDRANT_URL, port=QDRANT_PORT, https=True, api_key=qdrant_api_key)
openai_client = OpenAI(api_key=openai_api_key)

def generate_uuid(chunk):
    hash_object = hashlib.md5(chunk.encode())
    return str(uuid.UUID(hash_object.hexdigest()))

def chunk_text(text, chunk_size, overlap):
    length = len(text)
    chunks = []
    start = 0
    while start < length:
        end = start + chunk_size
        if end > length:
            end = length
        chunks.append(text[start:end])
        start = end - overlap
        if end == length:
            break
    return chunks

st.title("TRYCK PÅ KNAPPEN FÖR ATT AI SKA VETA ALLT DET SENASTE FRÅN MEDIAWIKI")
if st.button("Refresh ai knowledge"):
    # Start session
    S = requests.Session()

    # Step 1: Retrieve login token
    params = {
        'action': "query",
        'meta': "tokens",
        'type': "login",
        'format': "json"
    }
    response = S.get(url=URL, params=params)
    data = response.json()
    login_token = data['query']['tokens']['logintoken']

    # Step 2: Login
    params = {
        'action': "login",
        'lgname': USERNAME,
        'lgpassword': PASSWORD,
        'lgtoken': login_token,
        'format': "json"
    }
    response = S.post(URL, data=params)
    assert response.json()['login']['result'] == 'Success', "Login failed!"

    # Step 3: Fetch all articles
    params = {
        'action': "query",
        'list': "allpages",
        'aplimit': "1000",
        'format': "json"
    }
    response = S.get(URL, params=params)
    pages = response.json()['query']['allpages']

    # Step 4: Retrieve content for each page and format accordingly
    all_pages_content = []
    for page in pages:
        pageid = str(page['pageid'])
        params = {
            'action': "query",
            'format': "json",
            'prop': "revisions",
            'rvprop': "content|timestamp",
            'rvslots': "main",
            'pageids': pageid
        }
        response = S.get(URL, params=params)
        page = response.json()['query']['pages'][pageid]
        page_content = page['revisions'][0]['slots']['main']['*']
        page_title = page['title']
        last_revised = page['revisions'][0]['timestamp']
        
        all_pages_content.append({"title": page_title, "content": page_content, "last_revised":last_revised, "link_to_article": f'https://wiki01.utvecklingfalkenberg.se/index.php/{page_title.replace(" ", "_")}'})


    # DELETE COLLECTION
    qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
    vectors_config = VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    qdrant_client.recreate_collection(collection_name=COLLECTION_NAME, vectors_config=vectors_config)


    all_chunks = []
    for item in all_pages_content:
        text_chunks = chunk_text(item['content'], 4500, 500)
        num_chunks = len(text_chunks)
        for index, chunk in enumerate(text_chunks):
            chunk_data = {
                'title': item['title'],
                'chunk': f'Title: {item["title"]} Content: {chunk}',
                'chunk_info': f'Chunk {index + 1} of {num_chunks}',
                'link_to_article': item['link_to_article'],
                'last_revised': item['last_revised']
            }
            all_chunks.append(chunk_data)


    upserted_documents_count = 0
    for batch_start in range(0, len(all_chunks), BATCH_SIZE):
        batch_data = all_chunks[batch_start:batch_start + BATCH_SIZE]
        batch = [item['chunk'] for item in batch_data]
        response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        batch_embeddings = [e.embedding for e in response.data]


        for i, chunk in enumerate(batch):
            embeddings = batch_embeddings[i]
            doc_uuid = generate_uuid(chunk)
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[{
                    "id": doc_uuid,
                    "vector": embeddings,
                    "payload": batch_data[i]  # Directly use the structured data from batch_data
                }]
            )
            upserted_documents_count += 1
            if upserted_documents_count % 100 == 0:
                print(f"Upserted {upserted_documents_count} documents into '{COLLECTION_NAME}' collection.")
        time.sleep(SLEEP_TIME)

    st.success(f"Finished upserting. Total documents upserted: {upserted_documents_count}")
