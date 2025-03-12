import os
import sys

from helper.writer import convert_markdown_to_dict, write_json_formatted
from model.constant import DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_BEST_TEMPERATURE
from model.my_openai import content_generate_by_api
from model.prompt import EN_RE_PROMPT
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def evaluate_repetition(store_path, test_model_name, evaluate_model_name, chats_all, api_url=None, api_key=None, 
                        temperature=0.8, top_p=1, presence_penalty=0, frequency_penalty=0, max_tokens=1536)-> int:
    """
    Evaluates the repetitiveness of the generated chats.
    
    Returns:
        total_repetitive_score (int): Sum of the repetitive scores.
        suspend (bool): Flag set to True if any chat is off (repetitive score equals 1).
    """
    total_repetitive_score = 0
    for re_index, element in enumerate(chats_all, start=1):
        if api_url is None or api_key is None:
            # use the local model to evaluate the text quality
            print("use the local model to evaluate the text quality")
        else:
            if "deepseek" in evaluate_model_name:
                text_grading = [{"role": "user", "content": EN_RE_PROMPT+element}]
                re_grading_result = content_generate_by_api(evaluate_model_name, DEEPSEEK_API_URL, DEEPSEEK_API_KEY, text_grading, temperature=DEEPSEEK_BEST_TEMPERATURE, top_p=top_p, max_tokens=max_tokens)
            else:
                text_grading = [{"role": "system", "content": EN_RE_PROMPT}, {"role": "user", "content": element}]
                re_grading_result = content_generate_by_api(evaluate_model_name, api_url, api_key, text_grading, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        re_grading_result_dict = convert_markdown_to_dict(re_grading_result)
        repetition_file = f"{store_path}/repetitive/repetitive_grading-{re_index}-{temperature}-{top_p}-{presence_penalty}-{frequency_penalty}-{max_tokens}.json"
        write_json_formatted(re_grading_result_dict, repetition_file, indent=4, sort_keys=False)
        repetitive_score = re_grading_result_dict["repetitive_score"]
        if repetitive_score == 1:
            print(f"repetitive_score: {repetitive_score} is so bad, no need to test, skip to the next temperature round")
            return total_repetitive_score
        
        total_repetitive_score += repetitive_score

    return total_repetitive_score