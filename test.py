from llama_index.llms.gemini import Gemini


resp = Gemini(api_key=key).complete("Write a poem about a magic backpack")
print(resp)