import transformers
import torch
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ["TRANSFORMERS_CACHE"] = "/var/scratch/lsamson/LLMS"

class LLaMa:
    """
    A simple class to interact with ChatGPT models via the OpenAI API.
    """

    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", temperature: float = 0., max_new_tokens: int = 512):
        """
        Constructor for the ChatGPT class.

        Args:
            model_name (str): The ChatGPT model to use (e.g., 'gpt-3.5-turbo' or 'gpt-4').
            temperature (float): Sampling temperature for text generation.
            max_new_tokens (int): Maximum number of new tokens to generate in the response.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        if "Llama-2" in self.model_name:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.use_default_system_prompt = True
        else:
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model_name,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
                token="TODO",
            )

    def predict(self, prompt: str) -> str:
        """
        Send a prompt to the ChatGPT model and return its response.

        Args:
            prompt (str): The user prompt or query.

        Returns:
            str: The text response from the model.
        """
        if "Llama-3" in self.model_name:
            prompt = [
                {"role": "user", "content": prompt},
            ]
            outputs = self.pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,

            )
            return outputs[0]["generated_text"][-1]['content']

        else:
            conversation = []
            conversation.append({"role": "user", "content": prompt})

            input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt")

            input_ids = input_ids.to(self.model.device)

            generate_kwargs = dict(
                {"input_ids": input_ids},
                max_new_tokens=self.max_new_tokens,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
            )
            outputs = self.model.generate(**generate_kwargs)
            prompt_length = input_ids.shape[-1]

            # Slice the generated tokens to remove the prompt part
            generated_tokens = outputs[0][prompt_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return response


def remove_substring(string, substring):
    return string.replace(substring, "")

def cut_off_text(text, prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index != -1:
        return text[:index]
    else:
        return text

# Example usage:
if __name__ == "__main__":
    # Instantiate a LLaMA model
    llama_model = LLaMa(model_name="meta-llama/Llama-2-7b-chat-hf", temperature=0.0, max_new_tokens=512)

    # Create a simple prompt
    prompt_text = "Hello, how are you?"

    # Get the response from the LLaMA model
    response_text = llama_model.predict(prompt_text)

    # Print the response
    print("LLaMa Response:", response_text)
