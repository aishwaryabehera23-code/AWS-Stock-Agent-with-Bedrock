import streamlit as st
import boto3
import json

modelId = "amazon.nova-micro-v1:0"

st.set_page_config(page_title="AWS First GenAI Journey", page_icon="🤖")

st.title("AWS First GenAI Journey")

def call_bedrock_stream(prompt):
    bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    
    prompt_config = {
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 1000,
            "temperature": 0.7,
            "topP": 0.9
        }
    }
    
    response = bedrock.invoke_model_with_response_stream(
        body=json.dumps(prompt_config),
        modelId=modelId,
        accept="application/json",
        contentType="application/json"
    )
    
    stream = response.get("body")
    if stream:
        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                chunk_obj = json.loads(chunk.get("bytes").decode())
                if "contentBlockDelta" in chunk_obj:
                    text = chunk_obj["contentBlockDelta"]["delta"].get("text", "")
                    if text:
                        yield text

if "home_messages" not in st.session_state:
    st.session_state.home_messages = [
        {"role": "assistant", "content": "Hello! How can I assist you today?"}
    ]

for message in st.session_state.home_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your message"):
    st.session_state.home_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = call_bedrock_stream(prompt)
        full_response = st.write_stream(response)
        
    st.session_state.home_messages.append({"role": "assistant", "content": full_response})