import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.my_openai import content_generate_by_api
from model.constant import DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_BEST_TEMPERATURE
from helper.writer import chat_object, convert_markdown_to_dict, write_json_formatted
from model.Local.webui import generate_content_with_webui
from model.prompt import CHARACTER_SYSTEM_PROMPT, CN_TEXT_PROMPT, FIRST_MESSAGE

WORD_LIMIT = 260

def generate_save_chat(text_rounds, store_path, test_model_name, temperature, headers, user_prompts, sillytavern_url, sillytavern_api_url, top_p = 1, presence_penalty = 0, 
                       frequency_penalty = 0, max_tokens = 1536, first_message = FIRST_MESSAGE):
    """
    Generates chat rounds and saves the chat outputs.
    
    Returns:
        chats_all (list): A list of final chat string outputs.
        word_low (bool): Flag set to True if any chat did not reach the word_limit.
        total_word_count (int): Total word count across evaluated segments.
    """
    chats_all = []
    word_low = False
    total_word_count = 0

    for chat_index in range(1, text_rounds+1):     
        # Generate multi-turn chat from the model.
        chats = generate_content_with_webui(
            sillytavern_url, headers, sillytavern_api_url, 
            CHARACTER_SYSTEM_PROMPT, user_prompts, chats=[], 
            temperature=temperature, top_p=top_p, presence_penalty=presence_penalty, 
            frequency_penalty=frequency_penalty, max_tokens=max_tokens, first_message=first_message
        )
        # Check each targeted chat (skip some introductory turns)
        for chat in chats[3:len(user_prompts)*2+2:2]:
            total_word_count += len(chat["content"])
            if len(chat["content"]) < WORD_LIMIT:
                word_low = True 
        
        # Use last two turn outputs for further evaluation
        chat_three = [chat["content"] for chat in chats[-2:]]
        chat_string_three = "".join(chat_three)
        chats_all.append(chat_string_three)
        
        chat_json = chat_object(chats, temperature, top_p, presence_penalty, frequency_penalty, max_tokens)
        chat_file = f"{store_path}/chat/chat-{chat_index}-{temperature}-{top_p}-{presence_penalty}-{frequency_penalty}-{max_tokens}.json"
        write_json_formatted(chat_json, chat_file, indent=4, sort_keys=False)
    
    return chats_all, word_low, total_word_count


def evaluate_text_quality(store_path, test_model_name, evaluate_model_name, chats_all, api_url=None, api_key=None, temperature=0.8, top_p=1, presence_penalty=0, frequency_penalty=0, max_tokens=1536):
    total_text_score = 0
    total_advanced_text_score = 0
    text_record = []

    for text_index, element in enumerate(chats_all, start=1):
        if api_url is None or api_key is None:
            # use the local model to evaluate the text quality
            print("use the local model to evaluate the text quality")
        else:
            if "deepseek" in evaluate_model_name:
                text_grading = [{"role": "user", "content": CN_TEXT_PROMPT+element}]
                text_grading_result = content_generate_by_api(evaluate_model_name, DEEPSEEK_API_URL, DEEPSEEK_API_KEY, text_grading, temperature=DEEPSEEK_BEST_TEMPERATURE, top_p=top_p, max_tokens=max_tokens)
            else:
                text_grading = [{"role": "system", "content": CN_TEXT_PROMPT}, {"role": "user", "content": element}]
                text_grading_result = content_generate_by_api(evaluate_model_name, api_url, api_key, text_grading, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        text_grading_result_dict = convert_markdown_to_dict(text_grading_result)
        text_score = text_grading_result_dict["text_score"]
        advanced_text_score = text_grading_result_dict["advanced_text_score"]
        write_json_formatted(text_grading_result_dict, f"{store_path}/text/text_grading{text_index}-{temperature}-{top_p}-{presence_penalty}-{frequency_penalty}-{max_tokens}.json", indent=4, sort_keys=False)
        if text_score <= 2 or advanced_text_score <= 1:
            print(f"text_score: {text_score} or advanced_text_score: {advanced_text_score} is too low, skipping further evaluation.")
            return 0, 0

        total_text_score += text_score
        total_advanced_text_score += advanced_text_score
        

    return total_text_score, total_advanced_text_score