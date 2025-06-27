import torch
from typing import Type
from torch import nn
from dataset import TextDataset


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Create necessary layers
        """
        self.embedding = nn.Embedding(self.vocab_size, embed_size, padding_idx=dataset.pad_id)
        self.rnn = rnn_type(input_size=embed_size, hidden_size=hidden_size, num_layers=rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        # This is a placeholder, you may remove it.
        # logits = torch.randn(
        #     indices.shape[0], indices.shape[1], self.vocab_size,
        #     device=indices.device
        # )
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """
        emb = self.embedding(indices)
        output, _ = self.rnn(emb)
        logits = self.linear(output)
        logits = logits[:, :lengths.max(), :]
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        # This is a placeholder, you may remove it.
        # generated = prefix + ', а потом купил мужик шляпу, а она ему как раз.'
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """
        device = next(self.parameters()).device
        # Кодируем префикс с BOS
        prefix_ids = self.dataset.text2ids(prefix) if prefix else []
        if isinstance(prefix_ids, list) and len(prefix_ids) > 0 and isinstance(prefix_ids[0], list):
            prefix_ids = prefix_ids[0]
        input_ids = [self.dataset.bos_id] + (prefix_ids if prefix else [])
        input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(device)
        # Получаем скрытое состояние после префикса
        emb = self.embedding(input_tensor)
        output, hidden = self.rnn(emb)
        generated_ids = input_ids.copy()
        for _ in range(len(input_ids), self.max_length):
            last_token = torch.LongTensor([generated_ids[-1]]).unsqueeze(0).to(device)
            emb = self.embedding(last_token)
            output, hidden = self.rnn(emb, hidden)
            logits = self.linear(output[:, -1, :])
            logits = logits / temp
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated_ids.append(next_token)
            if next_token == self.dataset.eos_id:
                break
        # Декодируем (без BOS)
        # Обрезаем по EOS, если он есть
        if self.dataset.eos_id in generated_ids:
            eos_pos = generated_ids.index(self.dataset.eos_id)
            decode_ids = generated_ids[1:eos_pos]
        else:
            decode_ids = generated_ids[1:]
        generated = prefix + self.dataset.ids2text(decode_ids)
        if prefix and not generated.startswith(prefix):
            generated = prefix + ' ' + self.dataset.ids2text(decode_ids)
        return generated
