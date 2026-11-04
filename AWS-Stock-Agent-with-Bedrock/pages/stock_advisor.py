import streamlit as st
import boto3
import json
import os
from dotenv import load_dotenv

load_dotenv()

modelId = "amazon.nova-micro-v1:0"

st.set_page_config(page_title="Tra cứu thông tin chứng khoán", page_icon="📈")
st.markdown(
    """
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Tra cứu thông tin chứng khoán")

def generate_response(prompt):
    bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    
    prompt_config = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": f"You are a helpful stock and financial advisor. Provide actionable insights and market analysis for this query: {prompt}"
                    }
                ]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 1500,
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

if "advisor_messages" not in st.session_state:
    st.session_state.advisor_messages = []

for message in st.session_state.advisor_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Nhập câu hỏi của bạn về chứng khoán..."):
    st.session_state.advisor_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = generate_response(prompt)
        full_response = st.write_stream(response)
        
    st.session_state.advisor_messages.append({"role": "assistant", "content": full_response})