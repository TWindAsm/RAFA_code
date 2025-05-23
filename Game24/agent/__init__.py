import os
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, wait_random_exponential, stop_after_attempt
from groq import Groq

completion_tokens = prompt_tokens = 0
client = Groq(api_key="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")

@retry(retry=retry_if_exception_type(Exception), 
       wait=wait_random_exponential(min=1, max=60), 
       stop=stop_after_attempt(5))
def completions_with_backoff(**kwargs):
    if "prompt" in kwargs:
        #print("prompt was in kwargs, uses completions")
        return client.completions.create(**kwargs)
    else:
        assert "messages" in kwargs, "Either prompt or messages must be provided"
        #print("uses chat completions")
        return client.chat.completions.create(**kwargs)

def gpt_with_history(prompt, history, model="llama-3.3-70b-versatile", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = []
    #print("prompt in gpt with history: ", prompt)
    #print("history: ", history)
    for h in history:
        if 'answer' in h:
            messages.append({"role": "assistant", "content": h["answer"]})
            #print("answer was in history")
        if 'feedback' in h:
            messages.append({"role": "user", "content": h["feedback"]})
            #print("feedback was in history")
    messages.append({"role": "user", "content": prompt})
    #print("messages in gpt with history: ", messages)
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def gpt(prompt, model="llama-3.3-70b-versatile", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def chatgpt(messages, model="llama-3.3-70b-versatile", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        #print("cnt: ", cnt)
        n -= cnt
        #print("n after -= cnt: ", n)
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        #print("res: ", res)
        outputs.extend([choice.message.content for choice in res.choices])
        #print("outputs: ", outputs)
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    #print("outputs in chatgpt: ", outputs)
    return outputs

def gpt_usage(backend="llama-3.3-70b-versatile"):
    global completion_tokens, prompt_tokens
    if backend == "llama-3.3-70b-versatile":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    else:
        cost = completion_tokens / 1000 * 0.02 + prompt_tokens / 1000 * 0.02
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}

class Agent:
    def __init__(self):
        pass

    def act(self, env, obs):
        raise NotImplementedError
