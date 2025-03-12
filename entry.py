import os
import logging
import numpy as np
from dotenv import load_dotenv
from evaluation.logic_evaluation import evaluate_logic_quality, generate_save_logic_answer
from evaluation.repetition_evaluation import evaluate_repetition
from evaluation.text_evaluation import WORD_LIMIT, evaluate_text_quality, generate_save_chat
from helper.writer import anwser_object, convert_markdown_to_dict, write_json_formatted
from model.Local.webui import SILLYTAVERN_URL, WEBUI_API_URL
from model.constant import DEEPSEEK_API_KEY, DEEPSEEK_API_URL
from model.my_openai import content_generate_by_api
from model.prompt import BATCH_LOGIC_FIRST_PROMPT, BATCH_LOGIC_FIRST_PROMPT_JUDGE, BATCH_LOGIC_SECOND_PROMPT, BATCH_LOGIC_SECOND_PROMPT_JUDGE, CHARACTER_SYSTEM_PROMPT, EN_RE_PROMPT, CN_TEXT_PROMPT, FIRST_MESSAGE 
from helper.writer import chat_object


# model for evaluation
DEEPSEEK_MODEL_NAME = "deepseek-reasoner"

WORD_LIMIT_BUFFER = 60


load_dotenv()

logic_prompts = [
    BATCH_LOGIC_FIRST_PROMPT, 
    BATCH_LOGIC_SECOND_PROMPT
]
logic_grading_prompts = [
    BATCH_LOGIC_FIRST_PROMPT_JUDGE,
    BATCH_LOGIC_SECOND_PROMPT_JUDGE
]
TEXT_THRESHOLD = 0.8
ADVANCED_TEXT_THRESHOLD = 0.6

logic_low_threshold = 0.6
repetitive_high_threshold = 0.8
repetitive_low_threshold = 0.6
repetitive_perfect_score = 3
text_prefect_score = 5
advanced_text_prefect_score = 4

def init_test(test_model_name, evaluate_model_name, store_chat_path, temperature_high, temperature_low, temperature_step, headers, 
        user_prompts:list[str], text_rounds=5,logic_rounds=10, sillytavern_url = SILLYTAVERN_URL, sillytavern_api_url = WEBUI_API_URL, 
        evaluate_api_url = DEEPSEEK_API_URL, evaluate_api_key = DEEPSEEK_API_KEY):
    # create the store_chat_path if not exists
    if not os.path.exists(store_chat_path):
        os.makedirs(store_chat_path, exist_ok=True)
        sub_path = ["chat", "text", "logic", "repetitive"]
        for sub_path in sub_path:
            os.makedirs(os.path.join(store_chat_path, sub_path), exist_ok=True)

    temperature_range = np.arange(temperature_high, temperature_low, temperature_step).tolist()
    for temperature in temperature_range:
        presence_penalty = 0
        frequency_penalty = 0
        penalty_max_retries = 20
        logic_pass = True
        for retry in range(penalty_max_retries):
            # generate chats
            chats, word_low, total_word_count = generate_save_chat(text_rounds, store_chat_path, test_model_name, temperature, headers, user_prompts, 
            sillytavern_url, sillytavern_api_url, presence_penalty=presence_penalty, frequency_penalty=frequency_penalty)
            if word_low or total_word_count < text_rounds*(WORD_LIMIT+WORD_LIMIT_BUFFER):
                print(f"word_low: {word_low}, total_word_count: {total_word_count}")
                break
            # evaluate the text quality
            total_text_score, total_advanced_text_score = evaluate_text_quality(store_chat_path, test_model_name, evaluate_model_name, chats, 
            api_url=evaluate_api_url, api_key=evaluate_api_key, temperature=temperature, top_p=1, presence_penalty=presence_penalty, frequency_penalty=frequency_penalty, max_tokens=1536)
            if total_text_score < text_prefect_score*text_rounds*TEXT_THRESHOLD or total_advanced_text_score < advanced_text_prefect_score*text_rounds*ADVANCED_TEXT_THRESHOLD:
                print(f"temperature: {temperature}, total_text_score: {total_text_score}, total_advanced_text_score: {total_advanced_text_score}, failed to pass text evaluation.")
                break
            # generate logic answers and evaluate the logic quality
            logic_answers = generate_save_logic_answer(logic_rounds, store_chat_path, test_model_name, headers, logic_prompts, sillytavern_url, sillytavern_api_url, temperature=temperature, presence_penalty=presence_penalty, frequency_penalty=frequency_penalty)
            correct_answer_number_list = evaluate_logic_quality(store_chat_path, test_model_name, evaluate_model_name, logic_answers, logic_grading_prompts, api_url=evaluate_api_url, api_key=evaluate_api_key, temperature=temperature, presence_penalty=presence_penalty, frequency_penalty=frequency_penalty)
            for correct_answer_number in correct_answer_number_list:
                if correct_answer_number < logic_rounds*logic_low_threshold:
                    logic_pass = False
                    break
            if not logic_pass:
                print(f"temperature: {temperature}, failed to pass logic evaluation.")
                break
            # evaluate the repetitive score
            repetitive_score = evaluate_repetition(store_chat_path, test_model_name, evaluate_model_name, chats, api_url=evaluate_api_url, api_key=evaluate_api_key, temperature=temperature, presence_penalty=presence_penalty, frequency_penalty=frequency_penalty)
            if repetitive_low_threshold*repetitive_perfect_score*len(chats) <= repetitive_score < repetitive_high_threshold*repetitive_perfect_score*len(chats):
                presence_penalty += 0.05
                frequency_penalty += 0.05
            elif repetitive_score >= repetitive_high_threshold*repetitive_perfect_score*len(chats):
                # log the best parameters result with logging and store the result in a file
                logging.info(f"qualified parameters: temperature={temperature}, presence_penalty={presence_penalty}, frequency_penalty={frequency_penalty}")
                logging.info(f"qualified result: repetitive_score={repetitive_score}, text_score={total_text_score}, advanced_text_score={total_advanced_text_score}, logic_score={correct_answer_number_list}")
                with open(f"{store_chat_path}/best_result.txt", "w") as f:
                    f.write(f"qualified parameters: temperature={temperature}, presence_penalty={presence_penalty}, frequency_penalty={frequency_penalty}")
                    f.write(f"qualified result: repetitive_score={repetitive_score}, text_score={total_text_score}, advanced_text_score={total_advanced_text_score}, logic_score={correct_answer_number_list}")
                break


if __name__ == "__main__":
    test_model_name = "Peach-9B-8k-Roleplay_Q8_0"
    store_chat_path = f"/home/emerson/AI/LLM/frone_end/SillyTavern/my_dataset/{test_model_name}"
    headers = {  
    "Content-Type": "application/json",
    "Cookie": "X-CSRF-Token=60e3cd23c64cca120cfdb97f37356adcc56eb7180f579d5aa0676de8e9a81d81; session-8be85eb1=eyJjc3JmVG9rZW4iOiI0ZWYzZThiMGVlMTU5ZWQyZDNlZTE1MDViM2YzYjJjYmRkNmZlODM5Y2Q3MWNlNDk1NGM2NTlhMmI3MWU4ODA2In0=; session-8be85eb1.sig=EbkLnVpAM9OuFBZaEESrYw6pIIU",
        "X-CSRF-Token": "4ef3e8b0ee159ed2d3ee1505b3f3b2cbdd6fe839cd71ce4954c659a2b71e8806"
    }
    user_prompts = [
        '"刚刚还没有，不过现在有了", 我笑着对她说到，并极力掩盖着自己的疲惫,展示出我好的一面',
        '"不过我还是很好奇，不是说下个月来吗，为什么提前来了，我的房间还一团遭呢"',
        '“当着面看你，才发现你真的美，啊，你的美简直是毒药，我喘不过气来了，救命!” 随后我便躺到床上不动了。'
    ]
    init_test(test_model_name, DEEPSEEK_MODEL_NAME, store_chat_path, 0.85, 0.74, -0.05, headers, user_prompts, text_rounds=3, logic_rounds=3)
    print("done")