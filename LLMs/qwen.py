import transformers
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


# os.environ["TRANSFORMERS_CACHE"] = "/var/scratch/lsamson/LLMS"

class Qwen:
    """
    A simple class to interact with ChatGPT models via the OpenAI API.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", temperature: float = 0., max_new_tokens: int = 512):
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


        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            token="TODO",
            attn_implementation="flash_attention_2",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, prompt: str) -> str:
        """
        Send a prompt to the ChatGPT model and return its response.

        Args:
            prompt (str): The user prompt or query.

        Returns:
            str: The text response from the model.
        """

        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


# Example usage:
if __name__ == "__main__":
    # Instantiate a LLaMA model
    qwen_model = Qwen(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", temperature=0.0, max_new_tokens=50)

    # Create a simple prompt
    prompt_text = "Hello, how are you?"

    # Get the response from the Qwen model
    response_text = qwen_model.predict(prompt_text)

    # Print the response
    print("Qwen Response:", response_text)
