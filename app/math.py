import torch


class Math:

    def __init__(self, tknz, model):
        self.tknz = tknz
        self.model = model

    def sentence_vector(self, text):
        # 文章ベクトル作成
        max_length = 256
        print(text + ":vector creating")

        encoding = self.tknz(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        encoding = {k: v.clone().detach() for k, v in encoding.items()}
        attention_mask = encoding['attention_mask']

        with torch.no_grad():
            output = self.model(**encoding)
            last_hidden_state = output.last_hidden_state
            averaged_hidden_state = (
                last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)

        print(text + ":vector created")
        return averaged_hidden_state[0].cpu().to(torch.float64).detach().numpy().copy()
