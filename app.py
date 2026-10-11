import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from pprint import pprint
import uuid
import chromadb
import re
import requests
import random
import json


#-----------------------------------------
# Setup
#-----------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise Exception("API Key is missing")
else:
    print("Key found")
client = OpenAI()

#-----------------------------------------
# Document
#-----------------------------------------
import sys
# On Render, private_documents.py is uploaded as a Secret File and mounted here
sys.path.append("/etc/secrets")
from private_documents import (
    document_overview,
    document_education,
    document_professional_experience,
    document_skills,
    document_projects,
    document_interests,
    system_message,
)


#-----------------------------------------
# Chunking Function
#-----------------------------------------
# Chunk the document

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split `text` into overlapping chunks of up to `chunk_size` characters,
    each ending on a natural boundary where possible.
    """

    def find_cut_point(start: int, tentative_end: int) -> int:
        """
        Where to end the chunk inside text[start:tentative_end]: prefers a
        paragraph break, then newline, then sentence end, then space - but
        only if that boundary is past the halfway point of the chunk.
        Note: no `text`/`chunk_size` params needed - it reads them directly
        from the enclosing chunk_text() call.
        """
        halfway = start + chunk_size // 2
        window = text[start:tentative_end]

        for pattern in (r"\n\s*\n", r"\n", r"[.!?]\s", r" "):
            matches = list(re.finditer(pattern, window))
            if matches and start + matches[-1].end() > halfway:
                return start + matches[-1].end()

        return tentative_end

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            end = find_cut_point(start, end)

        chunks.append(text[start:end].strip())

        if end >= text_length:
            break
        start = end - overlap

    return chunks


#-----------------------------------------
# RAG: Chunk, Embed & Store in ChromaDB
#-----------------------------------------
documents = [
    {"text": document_overview, "source":"Overview"},
    {"text": document_professional_experience, "source":"Professional Experience"},
    {"text": document_education, "source":"Education"},
    {"text": document_skills, "source":"Skills"},
    {"text": document_projects, "source":"Projects"},
    {"text": document_interests, "source":"Interests"}
]
chunks = []
ids = []
metadatas = []

for doc in documents:
    # Prepare the lists
    chunks_ = [f"[{doc['source']}] {c}" for c in chunk_text(doc["text"], chunk_size=700, overlap=80)]
    ids_ = [str(uuid.uuid4()) for _ in range(len(chunks_))]
    metadatas_ = [{"source": doc["source"], "chunk_index": i} for i in range(len(chunks_))]

    # Add to main lists
    chunks.extend(chunks_)
    ids.extend(ids_)
    metadatas.extend(metadatas_)
print (f"Created {len(chunks)} chunks:\n")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} (ID:{ids[i]}, Source: {metadatas[i]['source']}, Index: {metadatas[i]['chunk_index']}, length: {len(chunk)})")
    print(chunk)
    print()

# Generate embeddings for all chunks
response = client.embeddings.create(
    model = "text-embedding-3-small",
    input = chunks
)
embeddings = [item.embedding for item in response.data]

#Verify emebeddings for logs
print(f"Generated {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} dimensions")

# Initialize ChromaDB and Store Vectors
#Initialize ChromaDB Client (Persistent Storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db_twin")
collection = chroma_client.get_or_create_collection(name="digital_twin")
# Empty the collection before adding new data(for testing purposes) 
if collection.get()["ids"]:
    collection.delete(collection.get()["ids"])

# Adding data to ChromaDB
collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=chunks,
    metadatas=metadatas
)
pprint(collection.get())

#-----------------------------------------
# Tools
#-----------------------------------------
tools = []
pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"


#Create send_notification function

import requests

def send_notification(message: str):
    if pushover_user is None or pushover_token is None: ### Handling of potential missing credentials
        return f"Notification failed: Pushover not configured"
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)
    return f"Notification sent: {message}"

### Describe Pushover as an LLM Tool
send_notification_function = {
    "name": "send_notification",
    "description": "Sends a push notification to Pooja. Use this when:\
          1)Someone wants to get in touch with her, hire or collaborate\
        - ask for their name and contact details first then send the notification to Pooja with the name and contact details.\
            2) You don't know the answer to the question about Pooja - send AUTOMATICALLY without asking, include the question so she can add this info later.",
    "parameters": {
        "type": "object",
        "properties":{
            "message":{
                "type": "string", "description": "The notification message to send to the user's device"}
        },
        "required":["message"]
    }
}

### Add Pushover to the list of tools or the LLM
tools.append({"type":"function", "function":send_notification_function})

# Simulates rolling a single six-sided die
def dice_roll():
    result = random.randint(1,6)
    return result

# Describe function for the LLM
roll_dice_function = {
    "name": "dice_roll",
        "description": "Simulates rolling a six-sided die and returns the result. Use this when the user wants to roll a die for games, decisions, or random number generation.",
        "parameters": {
            "type": "object",
            "properties":{},
            "required":[]
        }   
}
# Add function to list of tools to LLM
tools.append({"type":"function", "function":roll_dice_function})



#-----------------------------------------
# Tool Handler
#-----------------------------------------
##Note: Function handles 2 or more tool calls but only if its the same tool

def handle_tool_call(tool_calls):

    tool_results = []
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        #print(f" Calling function{function_name}") # For future debugging

        if function_name == "send_notification":
            content = send_notification(args["message"])             #Actually send the notification, i.e. call the tool
            
        elif function_name == "dice_roll":
           content = f"Rolled: {dice_roll()}"
        #elif function_name == "insert_function_name_3":
        #    content = insert_function_name_3(args["message"])  
        # ........
        # 
        else:
            content = f"Unknown function: {function_name}" 

        tool_call_result ={
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call.id
        }

        tool_results.append(tool_call_result)
    return tool_results




#-----------------------------------------
# Main Response Function
#-----------------------------------------
def respond_ai(message, history):
        

    # RAG: Embed the query using the same model we used for the chunks to ensure compatibility
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[message]
        )
        query_embedding = response.data[0].embedding
    
        # RAG:  Search ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],  # also expects a list of embeddings
            n_results=5
        )
    
        # RAG: Stitch retrieved chunks together to create the context for the response
        context = "\n\nContext:\n" + "\n---\n".join(results["documents"][0])

        # Print logs for debugging
        print("\n========================================\n")
        print(f"User Message: \n{message}\n")
        print("****Retrieved Chunks:")
        for a,b in zip(results["documents"][0],results["metadatas"][0]):
            print("-------------------------------")
            print(f"<<Document {b['source']} -- Chunk {b['chunk_index']}>>\n{a}\n")
            
    
        # RAG: Update system message with context for this conversation turn
        system_message_enhanced = system_message + "\n\nContext:\n" + context
    

    # Build messages for this turn
        messages = [{"role": "system", "content": system_message_enhanced}] + history + [{"role": "user", "content": message}]

    # Call LLM
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools
        )
        message = response.choices[0].message

        while message.tool_calls:
                from pprint import pprint
                pprint(message.tool_calls)
        
                tool_result = handle_tool_call(message.tool_calls)
                messages.append(message)
                messages.extend(tool_result)
        
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=messages,
                    tools=tools
                )
                message = response.choices[0].message

        return message.content


#-----------------------------------------
# Launch Gradio
#-----------------------------------------
port = int(os.environ.get("PORT", 7860))
avatar = "my_photo.JPG" if os.path.exists("my_photo.JPG") else None
gr.ChatInterface(fn=respond_ai,
                 title="Pooja's Digital Twin",
                 chatbot=gr.Chatbot(avatar_images=(None, avatar)),
                 description="Chat with an AI version of Pooja Hari Ambrish. Ask about her education, experience, skills, projects, interests or just say Hi",
                 examples=["Hi there! Tell me about yourself","What's your educational background?", "Tell me about your professional experience", "What are your technical skills?", "What are you learning right now?", "What do you do for fun?"]
                 ).launch(inbrowser=True,server_name="0.0.0.0",server_port=port)