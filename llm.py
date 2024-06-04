import torch
import torch.nn.functional as F


from transformers import AutoTokenizer, AutoModelForCausalLM

class BaseLLM:
    def __init__(self, args):
        self.args = args
        self.model_name = args.llm
        
        print(f"initializing {self.model_name}\n\n")
        self.initialize_model()
        print("finish loading model\n")

    @abstractmethod
    def initialize_model(self):
        pass
    
    def generate_next_token_predictions(self, token_sequences):
        
        if hasattr(self, 'model'):
            outputs = self.model(input_ids=token_sequences, output_hidden_states=True)
            return outputs.hidden_states[-1]
        else:
            raise AttributeError('Model is not initialized properly')
        
    def generate_next_token_predictions_withfv(self, token_fv):
    
        if hasattr(self, 'model'):
            outputs = self.model(inputs_embeds=token_fv, output_hidden_states=True)
            return outputs.hidden_states[-1]
        else:
            raise AttributeError('Model is not initialized properly')
        
    def translate(self, batch_feature_vectors):
        
        if hasattr(self, 'model'):
            batch_size, seq_len, embedding_dim = batch_feature_vectors.shape
            
            logits = torch.matmul(batch_feature_vectors, self.embeddings.T)
            sfmx = torch.softmax(logits/self.args.temperature, dim=2)
            closest_tokens = torch.argmax(sfmx, dim=2)        
            latent_vector = torch.matmul(sfmx, self.embeddings)
            
            # del sfmx
            
            return latent_vector, logits, closest_tokens
        else:
            raise AttributeError('Model is not initialized properly')


class GPT2Model(BaseLLM):
    def initialize_model(self):
        
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)

        self.embeddings = self.model.lm_head.weight
        self.feature_dim = self.model.config.hidden_size
        self.vocab_len = self.model.config.vocab_size
        
        self.device = self.args.device
        self.model.to(self.device)
        
        for param in self.model.parameters():
            param.requires_grad = False
    

class LlamaModel(BaseLLM):
    def initialize_model(self):
        
        from transformers import LlamaTokenizer, LlamaForCausalLM

        llm = 'openlm-research/open_llama_3b_v2'
        self.tokenizer = LlamaTokenizer.from_pretrained(llm)
        self.model = LlamaForCausalLM.from_pretrained(llm)
        
        # self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        # self.model = LlamaForCausalLM.from_pretrained(self.model_name)
        
        self.embeddings = self.model.lm_head.weight
        self.feature_dim = self.model.config.hidden_size
        self.vocab_len = self.model.config.vocab_size
        
        self.device = self.args.device
        self.model.to(self.device)
        
        for param in self.model.parameters():
            param.requires_grad = False
            
class GemmaModel(BaseLLM):
    def initialize_model(self):
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        llm = "google/gemma-2b"
        self.tokenizer = AutoTokenizer.from_pretrained(llm)
        self.model = AutoModelForCausalLM.from_pretrained(llm)
        
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        self.embeddings = self.model.lm_head.weight
        self.feature_dim = self.model.config.hidden_size
        self.vocab_len = self.model.config.vocab_size
        
        self.device = self.args.device
        self.model.to(self.device)
        
        for param in self.model.parameters():
            param.requires_grad = False