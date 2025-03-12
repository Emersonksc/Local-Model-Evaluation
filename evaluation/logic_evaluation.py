import os
import sys

from helper.util import split_into_columns
from helper.writer import anwser_object, convert_markdown_to_dict, write_json_formatted
from model.constant import DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_BEST_TEMPERATURE
from model.my_openai import content_generate_by_api
from model.prompt import BATCH_LOGIC_FIRST_PROMPT, BATCH_LOGIC_SECOND_PROMPT
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.Local.webui import  SILLYTAVERN_URL, WEBUI_API_URL, generate_content_with_webui



def generate_save_logic_answer(logic_rounds, store_path, model_name, headers, logic_prompts:list[str], 
                               sillytavern_url, sillytavern_api_url, temperature, top_p = 1, 
                              presence_penalty = 0, frequency_penalty = 0, max_tokens = 1536)->list[str]:
    """
    Generates logic answers and saves them.
    
    Returns:
        answers_first_all (list): A list of first logic answers.
        answers_second_all (list): A list of second logic answers.
    """    
    all_answers = []
    all_answers_string = []

    for logic_index in range(1, logic_rounds+1):
        for logic_prompt in logic_prompts:
            answer = generate_content_with_webui(
                sillytavern_url, headers, sillytavern_api_url, chats=[], user_prompts=[logic_prompt],
                temperature=temperature, top_p=top_p, presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty, max_tokens=max_tokens
            )
            all_answers.append(answer[1]["content"])
    split_answers = split_into_columns(all_answers, logic_rounds, len(logic_prompts)) 

    for logic_index, answers in enumerate(split_answers, start=1):
        answers_json = anwser_object(answers, temperature, top_p, presence_penalty, frequency_penalty, max_tokens)
        answer_file = f"{store_path}/logic/logic_answer-{logic_index}-{temperature}-{top_p}-{presence_penalty}-{frequency_penalty}-{max_tokens}.json"
        write_json_formatted(answers_json, answer_file, indent=4, sort_keys=False)
        all_answers_string.append("".join([f"{i+1}. {ans}\n" for i, ans in enumerate(answers)]))
        
    return all_answers_string


def evaluate_logic_quality(store_path, test_model_name, evaluate_model_name, all_answers_string:list[str], grading_prompts:list[str], api_url=None, api_key=None, 
                           temperature=0.8, top_p=1, presence_penalty=0, frequency_penalty=0, max_tokens=1536)->list[int]:
    correct_answer_number_list = []
    logic_index = 0
    for all_answer_str, grading_prompt in zip(all_answers_string, grading_prompts):
        logic_index += 1
        if api_url is None or api_key is None:
            # use the local model to evaluate the text quality
            print("use the local model to evaluate the text quality, currently not supported, program exit")
            sys.exit()
        else:
            if "deepseek" in evaluate_model_name:
                text_grading = [{"role": "user", "content": grading_prompt+all_answer_str}]
                text_grading_result = content_generate_by_api(evaluate_model_name, DEEPSEEK_API_URL, DEEPSEEK_API_KEY, text_grading, temperature=DEEPSEEK_BEST_TEMPERATURE, top_p=top_p, max_tokens=max_tokens)
            else:
                text_grading = [{"role": "system", "content": grading_prompt}, {"role": "user", "content": all_answer_str}]
                text_grading_result = content_generate_by_api(evaluate_model_name, api_url, api_key, text_grading, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        text_grading_result_dict = convert_markdown_to_dict(text_grading_result)
        correct_answer_number_list.append(text_grading_result_dict["total_correct_answer_number"])
        logic_file = f"{store_path}/logic/logic_grading-{temperature}-{top_p}-{presence_penalty}-{frequency_penalty}-{max_tokens}.json"
        write_json_formatted(text_grading_result_dict, logic_file, indent=4, sort_keys=False)


    return correct_answer_number_list