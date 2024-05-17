from torch.nn import Embedding as _Embedding


class Embedding(_Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dim = self.embedding_dim
