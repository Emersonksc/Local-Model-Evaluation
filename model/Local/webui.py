import random
import re
import sys
import time
import requests
import json
import logging

logger = logging.getLogger(__name__)
SILLYTAVERN_URL = "http://127.0.0.1:8000/api/backends/chat-completions/generate"
WEBUI_API_URL = "http://127.0.0.1:5000/v1"
# This is a function that simulates the sillytavern to generate content using the oobabooga web ui as the backend.
def request_sillytavern(url, headers, payload):
    try:
        # Use json=payload and set a timeout
        response = requests.post(url, headers=headers, json=payload, timeout=20)
    except requests.RequestException as e:
        logger.error("Request exception: %s", e)
        sys.exit(1)
        
    if response.status_code == 200:
        response_data = response.json()
        content = response_data.get("choices")[0].get("message").get("content")
        if isinstance(content, str):
            return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        else:
            logger.error("Content is not str: %s", content)
            sys.exit(1)
    else:
        logger.error("Request failed with status code: %s\nResponse Text: %s", response.status_code, response.text)
        sys.exit(1)

def generate_content_with_webui(sillytavern_generate_url, headers, api_url, system_prompt=None, user_prompts=[], chats=[], temperature=0.7, max_tokens=1568, presence_penalty=0, frequency_penalty=0, top_p=1, first_message=None):
    for user_prompt in user_prompts:
        time.sleep(2)
        if not chats:
            if system_prompt:
                chats.append({"role": "system", "content": system_prompt})
            if first_message:
                chats.append({"role": "assistant", "content": first_message})
            chats.append({"role": "user", "content": user_prompt})
        else:
            chats.append({"role": "user", "content": user_prompt})

        payload = {
            "messages": chats,
            "model": "1",
            "temperature": temperature,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_completion_source": "custom",
            "user_name": "user",
            "char_name": "character",
            "group_names": [],
            "show_thoughts": True,
            "custom_url": api_url,
            "custom_include_body": "",
            "custom_exclude_body": "",
            "custom_include_headers": "",
            "custom_prompt_post_processing": ""
        }
        content = request_sillytavern(sillytavern_generate_url, headers, payload)
        if content:
            chats.append({"role": "assistant", "content": content})
        else:
            logger.error("Request failed")
            sys.exit(1)

    return chats
        
def generate_content_with_webui_completion_api(sillytavern_generate_url, headers, api_url, system_prompt=None, user_prompts=[], chats=[], temperature=0.7, max_tokens=1568, presence_penalty=0, frequency_penalty=0, top_p=1, first_message=None):
    for user_prompt in user_prompts:
        time.sleep(2)
        if not chats:
            if system_prompt:
                chats.append({"role": "system", "content": system_prompt})
