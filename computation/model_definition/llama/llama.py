from transformers import LlamaForSequenceClassification, LlamaModel
from computation.model_definition.llama.self_attention import CustomLlamaAttention


class CustomLlamaModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        # Replace the standard attention layer with the custom attention layer
        for i in range(len(self.encoder.block)):
            self.encoder.block[i].attn = CustomLlamaAttention(config)

class CustomLlamaForSequenceClassification(LlamaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        # Use the custom model with the modified attention layers
        self.model = CustomLlamaModel(config)