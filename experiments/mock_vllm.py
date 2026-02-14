
# Mock vLLM for verification
class SamplingParams:
    def __init__(self, temperature=0.7, max_tokens=2048):
        pass

class LLM:
    def __init__(self, model, tensor_parallel_size=1, max_model_len=4096):
        pass
    
    def get_tokenizer(self):
        return None
        
    def generate(self, prompts, sampling_params):
        results = []
        for p in prompts:
            # Create a mock output object structure that matches vLLM
            class MockOutput:
                def __init__(self, prompt):
                    self.prompt = prompt
                    self.prompt_token_ids = [1] * (len(prompt) // 4) # Fake token count
                    
                    class OutputItem:
                        def __init__(self):
                            self.text = "Mock Answer"
                            self.token_ids = [2] * 50
                    self.outputs = [OutputItem()]
                    
            results.append(MockOutput(p))
        return results
