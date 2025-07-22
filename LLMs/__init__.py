

def init_llm(model_name, temperature, max_new_tokens):
    if "gpt" in model_name.lower() or "o3" in model_name.lower():
        from LLMs.chatgpt import ChatGPT
        return ChatGPT(model_name, temperature, max_new_tokens)
    elif "llama" in model_name.lower():
        from LLMs.llama import LLaMa
        return LLaMa(model_name, temperature, max_new_tokens)
    elif "phi" in model_name.lower():
        from LLMs.phi import Phi
        return Phi(model_name, temperature, max_new_tokens)
    elif "qwen" in model_name.lower():
        from LLMs.qwen import Qwen
        return Qwen(model_name, temperature, max_new_tokens)
    elif "deepseek" in model_name.lower():
        from LLMs.deepseek import DeepSeek
        return DeepSeek(model_name, temperature, max_new_tokens)
    elif "claude" in model_name.lower():
        from LLMs.claude import Claude
        return Claude(model_name, temperature, max_new_tokens)
    elif "gemma" in model_name.lower():
        from LLMs.gemma import Gemma
        return Gemma(model_name, temperature, max_new_tokens)
    elif "mistral" in model_name.lower():
        from LLMs.mistral import MistralAPI
        return MistralAPI(model_name, temperature, max_new_tokens)
    elif "gemini" in model_name.lower():
        from LLMs.gemini import Gemini
        return Gemini(model_name, temperature, max_new_tokens)
    elif "grok" in model_name.lower():
        from LLMs.grok import Grok
        return Grok(model_name, temperature, max_new_tokens)
    else:
        raise("LLM Model not Found")