import math
import torch
import torch.nn as nn
import torch.optim as optim


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + x + self.pe[:, :x.size(1)]
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        self.output_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x, num_heads):
        batch_size, seq_length, d_model = x.size()
        depth = d_model // num_heads
        return x.view(batch_size, seq_length, num_heads, depth).transpose(1, 2)

    def forward(self, query, key, value, d_model, num_heads, mask=None):
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        query = self.split_heads(query, num_heads)
        key = self.split_heads(key, num_heads)
        value = self.split_heads(value, num_heads)

        depth = d_model // num_heads
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(depth)

        if mask is not None:
            scores += scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)

        attention_output = torch.matmul(attention_weights, value)

        batch_size, _, seq_length, d_k = attention_output.size()
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, d_model)

        attention_output = self.output_linear(attention_output)
        return attention_output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadedAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, d_model, num_heads, mask):
        attention_output = self.self_attention(x, x, x, d_model, num_heads, mask)
        attention_output = self.dropout(attention_output)
        x = x + attention_output
        x = self.norm1(x)

        feed_forward_output = self.feed_forward(x)
        feed_forward_output = self.dropout(feed_forward_output)
        x = x + feed_forward_output
        x = self.norm2(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.masked_self_attention = MultiHeadedAttention(d_model, num_heads)
        self.enc_dec_attention = MultiHeadedAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, d_model, num_heads, src_mask, tgt_mask):
        self_attention_output = self.masked_self_attention(x, x, x, d_model, num_heads, tgt_mask)
        self_attention_output = self.dropout(self_attention_output)
        x = x + self_attention_output
        x = self.norm1(x)

        enc_dec_attention_output = self.enc_dec_attention(x, encoder_output, encoder_output, d_model, num_heads, src_mask)
        enc_dec_attention_output = self.dropout(enc_dec_attention_output)
        x = x + enc_dec_attention_output
        x = self.norm2(x)

        feed_forward_output = self.feed_forward(x)
        feed_forward_output = self.dropout(feed_forward_output)
        x = x + feed_forward_output
        x = self.norm3(x)
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, max_len, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_layer_1 = EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder_layer_2 = EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder_layer_1 = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder_layer_2 = DecoderLayer(d_model, num_heads, d_ff, dropout)

        self.linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        device = src.device
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).to(device)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=device), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt, d_model, num_heads):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        encoder_embedding = self.encoder_embedding(src)
        en_positional_encoding = self.positional_encoding(encoder_embedding)
        src_embedded = self.dropout(en_positional_encoding)

        decoder_embedding = self.decoder_embedding(tgt)
        de_positional_encoding = self.positional_encoding(decoder_embedding)
        tgt_embedded = self.dropout(de_positional_encoding)

        enc_output = src_embedded
        enc_output = self.encoder_layer_1(enc_output, d_model, num_heads, src_mask)
        enc_output = self.encoder_layer_2(enc_output, d_model, num_heads, src_mask)

        dec_output = tgt_embedded
        dec_output = self.decoder_layer_1(dec_output, enc_output, d_model, num_heads, src_mask, tgt_mask)
        dec_output = self.decoder_layer_2(dec_output, enc_output, d_model, num_heads, src_mask, tgt_mask)

        output = self.linear(dec_output)
        return output


def go():
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    num_heads = 8
    # num_layers = 2
    d_ff = 2048
    max_len = 50
    dropout = 0.1
    num_examples = 100
    seq_length = 10
    epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    # transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len,
    # dropout).to(device)
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, max_len, dropout).to(device)

    src_data = torch.randint(1, src_vocab_size, (num_examples, max_len)).to(device)
    tgt_data = torch.randint(1, tgt_vocab_size, (num_examples, max_len)).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    print(transformer(src_data, tgt_data[:, :-1], d_model, num_heads).shape)
    print(src_data)

    transformer.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(num_examples):
            src = src_data[i].unsqueeze(0)
            tgt = tgt_data[i].unsqueeze(0)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            optimizer.zero_grad()
            output = transformer(src, tgt_input, d_model, num_heads)

            loss = criterion(output.view(-1, tgt_vocab_size), tgt_output.contiguous().view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / num_examples}')

    transformer.eval()
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for i in range(num_examples):
            src = src_data[i].unsqueeze(0)
            tgt = tgt_data[i].unsqueeze(0)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = transformer(src, tgt_input, d_model, num_heads)

            predictions = torch.argmax(output, dim=-1)
            correct = (predictions == tgt_output).sum().item()
            total_correct += correct
            total_tokens += tgt_output.numel()

        accuracy = total_correct / total_tokens
        print(f'Accuracy: {accuracy * 100:.2f}%')
