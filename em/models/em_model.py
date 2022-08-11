import torch.nn as nn
from em.models.em_base_model import BaseModel, Highway, SimpleClassifier, RobertaClassificationHead, DeepSet


class EmModel(BaseModel):
    def __init__(self, pretrained_lm='bert', sent_size=256, freeze=True, deep_classifier=False):
        super(EmModel, self).__init__()

        self.lm_name = pretrained_lm
        self.sent_size = sent_size
        self.freeze = freeze
        self.context_similarity_layers(deep=deep_classifier)
        self.hw = Highway(2)
        self.out = nn.Linear(2, 2)
        self.use_pooled = False
        self.log_softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, concat):
        ctx_logits = self.context_forward(concat)
        output = self.out(self.hw(ctx_logits))

        return ctx_logits

    def context_similarity_layers(self, deep=False):
        self.bert = self.get_lm_class().from_pretrained(self.get_lm(), cache_dir="transformer_cache/")
        for param in self.bert.base_model.parameters():
            param.requires_grad = True

        if deep:
            self.ctx_classifier = RobertaClassificationHead(self.get_lm_dim())
            #self.ctx_classifier = DeepSet(self.get_lm_dim())
        else:
            self.ctx_classifier = SimpleClassifier(self.get_lm_dim())

    def resize_embedding(self, module, new_len):
        if module == 'ctx':
            self.bert.resize_token_embeddings(new_len)

    def context_forward(self, x):
        # 0: logits output,  1:pooled output
        if self.has_type_token():
            out = self.bert(x['input_ids'], attention_mask=x['attention_mask'], token_type_ids=x['token_type_ids'])
        else:
            out = self.bert(x['input_ids'], attention_mask=x['attention_mask'])

        if self.use_pooled:
            out = out[1]
        else:
            out = out[0][:, 0, :]

        out = self.ctx_classifier(out)

        return out