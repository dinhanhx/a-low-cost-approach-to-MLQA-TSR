import torch
from transformers import AutoModel


class TextEmbedding:
    def __init__(self) -> None:
        self.text_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        if torch.cuda.is_available():
            self.text_model.eval().cuda()

    def infer_single(self, text: str):
        with torch.no_grad():
            outputs = self.text_model.encode([text])

        torch.cuda.empty_cache()
        return outputs[0].tolist()
