import os
import re
import time
import traceback
from typing import Optional, Union
import httpx
import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.prompt import FIRST_MESSAGE
from openai import OpenAI, OpenAIError
from pydantic import BaseModel

logger = logging.getLogger(__name__)

def batch_generate_content(model_path: str, base_url: str, api_key: str, system_prompt: str, user_prompts: list[str],
                           chats: list[dict] = [],
                           response_format: Optional[BaseModel] = None, max_retries: int = 0,
                           temperature: float = 0.7,
                           top_p: float = 1.0,
                           presence_penalty: float = None,
                           frequency_penalty: float = None,
                           max_tokens: int = 1536,
                           ) -> list[dict]:
    for user_prompt in user_prompts:

        # judge if the chats is empty
        if len(chats) == 0:
            chats = [{"role": "system", "content": system_prompt}]
            chats.append({"role": "assistant", "content": FIRST_MESSAGE})
            chats.append({"role": "user", "content": user_prompt})
            assistant_message = content_generate_by_api(model_path, base_url, api_key, chats,
                                                 response_format, max_retries, temperature, top_p, presence_penalty, frequency_penalty, max_tokens)
            if isinstance(assistant_message, str):
                assistant_message_no_think = re.sub(r"<think>.*?</think>", "", assistant_message, flags=re.DOTALL)
                chats.append({"role": "assistant", "content": assistant_message_no_think})
            else:
                logger.error("Assistant message is not str: %s", assistant_message)
                sys.exit(1)
        else:
            chats.append({"role": "user", "content": user_prompt})
            assistant_message = content_generate_by_api(model_path, base_url, api_key, chats,
                                                 response_format, max_retries, temperature, top_p, presence_penalty, frequency_penalty, max_tokens)
            if isinstance(assistant_message, str):  
                assistant_message_no_think = re.sub(r"<think>.*?</think>", "", assistant_message, flags=re.DOTALL)
                chats.append({"role": "assistant", "content": assistant_message_no_think})
            else:
                logger.error("Assistant message is not str: %s", assistant_message)
                sys.exit(1)
    return chats


class Score(BaseModel):
    text_score: str
    text_comment: str


def generate_parsed_content(
        model_name: str,
        base_url: str,
        api_key: str,
        messages: list[dict],
        response_format: BaseModel,
        temperature: float = 0.7,
        top_p: float = 1.0,
        presence_penalty: float = None,
        frequency_penalty: float = None,
        max_tokens: int = 1536,
) -> Optional[BaseModel]:
    """
    Generate a parsed content from the model.
    """
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    params = {
        "model": model_name,
        "n": 1,
        "messages": messages,
        "response_format": response_format,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "max_tokens": max_tokens
    }
    params = {key: value for key, value in params.items() if value is not None}
    response = client.beta.chat.completions.parse(**params)
    message = response.choices[0].message
    return message


def generate_raw_content(
    model_name: str,
    base_url: str,
    api_key: str,
    messages: list[dict],
    temperature: float = 0.7,
    top_p: float = 1.0,
    presence_penalty: float = None,
    frequency_penalty: float = None,
    max_tokens: int = 1536,
) -> Optional[str]:
    """
    Generate a raw content from the model.
    """
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    params = {
        "model": model_name,
        "n": 1,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens,
        "frequency_penalty": frequency_penalty
    }
    params = {key: value for key, value in params.items() if value is not None}
    try:
        response = client.chat.completions.create(**params)
        message = response.choices[0].message.content
    except Exception as e:
        logger.error("Error during raw content generation: %s", e)
        raise
    logger.info(message)
    return message


def content_generate_by_api(
        model_name: str,
        base_url: str,
        api_key: str,
        messages: list[dict],
        response_format: Optional[BaseModel] = None,
        max_retries: int = 6,  # Change this value as needed
        temperature: float = 0.7,
        top_p: float = 1.0,
        presence_penalty: float = 0,
        frequency_penalty: Optional[float] = 0,
        max_tokens: int = 2048,
) -> Union[BaseModel, str, None]:
    """
    Generate content from the model while retrying on certain errors.
    Retries on 429 (rate limit/resource exhausted), 500 errors,
    and timeout/connection errors.
    """
    attempt = 0
    backoff_time = 1  # Initial wait before retrying
    while True:
        try:
            start_time = time.time()
            if response_format is None:
                return generate_raw_content(
                    model_name,
                    base_url,
                    api_key,
                    messages,
                    temperature,
                    top_p,
                    presence_penalty,
                    frequency_penalty,
                    max_tokens
                )
            else:
                return generate_parsed_content(
                    model_name,
                    base_url,
                    api_key,
                    messages,
                    response_format,
                    temperature,
                    top_p,
                    presence_penalty,
                    frequency_penalty,
                    max_tokens
                )
        except OpenAIError as oe:
            status = getattr(oe, "http_status", None)
            error_str = str(oe)
            # Check for a timeout error in the exception string
            if "Request timed out" in error_str or "Timeout" in error_str:
                attempt += 1
                if attempt > max_retries:
                    logger.error("Maximum retry attempts (%d) reached for timeout error.", max_retries)
                    raise oe
                logger.warning("Timeout error (OpenAI): %s. Attempt %d/%d. Retrying in %d seconds...", 
                               oe, attempt, max_retries, backoff_time)
                time.sleep(backoff_time)
                backoff_time *= 2
                continue
            # Check for rate limit error 429
            elif status == 429 or "429" in error_str:
                attempt += 1
                if attempt > max_retries:
                    logger.error("Maximum retry attempts (%d) reached for error 429.", max_retries)
                    raise oe
                logger.warning("Rate limit/resource exhaustion error (429): %s. Attempt %d/%d. Retrying in %d seconds...", 
                               oe, attempt, max_retries, backoff_time)
                time.sleep(backoff_time)
                backoff_time *= 2
                continue
            # Check for server error 500
            elif status == 500 or "500" in error_str:
                attempt += 1
                if attempt > max_retries:
                    logger.error("Maximum retry attempts (%d) reached for server error 500.", max_retries)
                    raise oe
                logger.warning("Server error (500): %s. Attempt %d/%d. Retrying in %d seconds...", 
                               oe, attempt, max_retries, backoff_time)
                time.sleep(backoff_time)
                backoff_time *= 2
                continue
            # Handle connection-related errors in OpenAI exceptions
            elif "Connection error" in error_str or "connection" in error_str.lower():
                attempt += 1
                if attempt > max_retries:
                    logger.error("Maximum retry attempts (%d) reached for connection error.", max_retries)
                    raise oe
                logger.warning("Connection error (OpenAI): %s. Attempt %d/%d. Retrying in %d seconds...", 
                               oe, attempt, max_retries, backoff_time)
                time.sleep(backoff_time)
                backoff_time *= 2
                continue
            else:
                logger.error("OpenAI error occurred: %s. No retry will be attempted for this error.", oe)
                raise oe
        except httpx.RemoteProtocolError as rpe:
            attempt += 1
            if attempt > max_retries:
                logger.error("Maximum retry attempts (%d) reached for RemoteProtocolError.", max_retries)
                raise rpe
            logger.warning("RemoteProtocolError: %s. Attempt %d/%d. Retrying in %d seconds...", 
                           rpe, attempt, max_retries, backoff_time)
            time.sleep(backoff_time)
            backoff_time *= 2
            continue
        except httpx.ConnectTimeout as cte:
            attempt += 1
            if attempt > max_retries:
                logger.error("Maximum retry attempts (%d) reached for ConnectTimeout error.", max_retries)
                raise cte
            logger.warning("ConnectTimeout error: %s. Attempt %d/%d. Retrying in %d seconds...", 
                           cte, attempt, max_retries, backoff_time)
            time.sleep(backoff_time)
            backoff_time *= 2
            continue
        except TimeoutError as te:
            attempt += 1
            if attempt > max_retries:
                logger.error("Maximum retry attempts (%d) reached for TimeoutError.", max_retries)
                raise te
            logger.warning("TimeoutError: %s. Attempt %d/%d. Retrying in %d seconds...", 
                           te, attempt, max_retries, backoff_time)
            time.sleep(backoff_time)
            backoff_time *= 2
            continue
        except httpx.ConnectError as ce:
            attempt += 1
            if attempt > max_retries:
                logger.error("Maximum retry attempts (%d) reached for ConnectError.", max_retries)
                raise ce
            logger.warning("ConnectError: %s. Attempt %d/%d. Retrying in %d seconds...", 
                           ce, attempt, max_retries, backoff_time)
            time.sleep(backoff_time)
            backoff_time *= 2
            continue
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                print("An exception occurred:", e)
                traceback.print_exc()
                logger.error("Maximum retry attempts (%d) reached for unexpected error.", max_retries)
                raise e
            logger.warning("Unexpected error: %s. Attempt %d/%d. Retrying in %d seconds...", 
                           e, attempt, max_retries, backoff_time)
            time.sleep(backoff_time)
            backoff_time *= 2
            continue

