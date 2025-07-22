import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import torch
import os

os.environ["TRANSFORMERS_CACHE"] = "/var/scratch/lsamson/LLMS"

class Phi:
    """
    A simple class to interact with ChatGPT models via the OpenAI API.
    """

    def __init__(self, model_name: str = "microsoft/phi-4", temperature: float = 0., max_new_tokens: int = 512):
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
        if "3" in self.model_name.lower():
            print("PHI-3")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            )
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )
        else:
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=self.model_name,
                model_kwargs={"torch_dtype": "auto"},
                device_map="auto",
        )



    def predict(self, prompt: str) -> str:
        """
        Send a prompt to the ChatGPT model and return its response.

        Args:
            prompt (str): The user prompt or query.

        Returns:
            str: The text response from the model.
        """

        messages = [
            {"role": "user", "content": prompt},
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature
        )
        return outputs[0]["generated_text"][-1]['content']


# Example usage:
if __name__ == "__main__":
    # Instantiate a Phi model
    phi_model = Phi(temperature=0.0, max_new_tokens=50)

    # Create a simple prompt
    prompt_text = "Hello, how are you?"

    # Get the response from the LLaMA model
    response_text = phi_model.predict(prompt_text)

    # Print the response
    print("Phi Response:", response_text)
