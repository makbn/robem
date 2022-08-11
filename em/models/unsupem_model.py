import torch
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from em.models.em_base_model import BaseModel


class UNSUPEmModel(BaseModel):
    def __init__(self, pretrained_lm='bert', sent_size=256):
        super(UNSUPEmModel, self).__init__()
        self.lm_name = pretrained_lm
        self.ctx_w = 1
        self.sent_size = sent_size
        self._context_similarity_layers()


    def _context_similarity_layers(self):
        self.bert = self.get_lm_class().from_pretrained(self.get_lm(), cache_dir="transformer_cache/")

        for param in self.bert.base_model.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        if self.has_type_token():
            ctx_1 = \
                self.bert(x1['input_ids'], attention_mask=x1['attention_mask'], token_type_ids=x1['token_type_ids'])[0][:, 0, :]
            ctx_2 = \
                self.bert(x2['input_ids'], attention_mask=x2['attention_mask'], token_type_ids=x2['token_type_ids'])[0][:, 0, :]
        else:
            ctx_1 = self.bert(x1['input_ids'], attention_mask=x1['attention_mask'])[0][:, 0, :]
            ctx_2 = self.bert(x2['input_ids'], attention_mask=x2['attention_mask'])[0][:, 0, :]

        ctx_sim = cosine_similarity(F.tanh(ctx_1), F.tanh(ctx_2), dim=1)
        sim = (self.ctx_w * ctx_sim)
        return sim
