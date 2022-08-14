from .architectures.set_encoder import SetEncoder
from torch import nn
import numpy as np
import torch
import pytorch_lightning as pl



class ETIN_model(pl.LightningModule):
    def __init__(self, cfg, info_for_model):
        super().__init__()
        assert info_for_model['max_variables'] is not None
        # He puesto + 2 porque necesitas embs para ini, pad
        self.padding_idx = info_for_model['padding_idx']
        language_size = info_for_model['language_size']
        self.tok_embedding = nn.Embedding(language_size + 2, cfg.dim_hidden, padding_idx=self.padding_idx)
        size_of_pos = (info_for_model['max_len'] + 1) * (info_for_model['memory_size'] + 1)
        self.pos_embedding = nn.Embedding(size_of_pos, cfg.dim_hidden)
        if cfg.sinusoidal_embeddings:
            self.create_sinusoidal_embeddings(
                size_of_pos, cfg.dim_hidden, out=self.pos_embedding.weight
            )
        dim_size_encoder = info_for_model['max_variables'] + info_for_model['memory_size'] + 1
        self.set_encoder = SetEncoder(cfg, dim_size_encoder)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.dim_hidden,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
        )
        self.decoder_transfomer = nn.TransformerDecoder(decoder_layer, num_layers=cfg.layers_decoder)
        self.fc_out = nn.Linear(cfg.dim_hidden, language_size)
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.padding_idx, reduction='none')
        self.dropout = nn.Dropout(cfg.dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.info_for_model = info_for_model
    

    def add_train_cfg(self, train_cfg):
        self.train_cfg = train_cfg


    def create_sinusoidal_embeddings(self, n_pos, dim, out):
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        out.requires_grad = False


    def forward(self, X, prev_exprs, expr=None, phase='eval'):
        # X -> B x N x X
        enc_src = self.set_encoder(X)
        # enc_src: B x F x E

        if phase == 'eval':
            input_decoder = prev_exprs.type(torch.int64)
        elif phase == 'train':
            input_decoder = torch.cat([prev_exprs, expr], dim=1).type(torch.int64)[:, :-1]
        else:
            raise ValueError('Phase must be either eval or train.')
        # input_decoder: B x T-1
        
        pos = self.pos_embedding(
            torch.arange(0, input_decoder.shape[1])
            .repeat(input_decoder.shape[0], 1)
            .type(torch.long).to(self.device)
        )

        te = self.tok_embedding(input_decoder)
        trg_ = self.dropout(te + pos)
        mask = nn.Transformer.generate_square_subsequent_mask(trg_.shape[1]).to(self.device)
        output = self.decoder_transfomer(
            trg_.permute(1, 0, 2),
            enc_src.permute(1, 0, 2),
            mask
        )
        output = self.fc_out(output).permute(1, 0, 2)
        if phase == 'eval':
            return self.softmax(output)
        return output


    def training_step(self, batch, _):
        X, prev_exprs, expr, for_loss = batch[0], batch[1], batch[2], batch[3]
        output = self.forward(X, prev_exprs, expr=expr, phase='train')
        loss = self.compute_loss(output, expr, for_loss)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    
    def compute_loss(self, output, trg, lengths):
        output = output[:, (output.shape[1] - self.info_for_model['max_len']):, :]
        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg.contiguous().view(-1).type(torch.long)
        loss = self.criterion(output, trg)
        weights = (torch.ones(loss.shape) * self.train_cfg.weight_length).to(self.device)
        lengths = lengths.repeat_interleave(self.info_for_model['max_len']).to(self.device)
        weights = torch.pow(weights, lengths)
        loss = torch.mul(loss, weights).sum() / (trg != self.padding_idx).sum()
        return loss


    def validation_step(self, batch, _):
        X, prev_exprs, expr, for_loss = batch[0], batch[1], batch[2], batch[3]
        output = self.forward(X, prev_exprs, expr=expr, phase='train')
        loss = self.compute_loss(output, expr, for_loss)
        self.log("val_loss", loss, on_epoch=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.train_cfg.lr)
        return optimizer



def create_input(row, language):
    # Input Dataset -> Meter aqui tambien los errores
    # X: ? x ? -> N x V
    padding = language.max_variables - row['X'].shape[1]
    padding_x = torch.nn.ConstantPad1d((0, padding), 0)
    X = padding_x(torch.Tensor(row['X']))
    # y: ? -> N x 1
    y = torch.Tensor(row['y']).unsqueeze(1)
    # y_preds ? x ? -> N x M  # Cambiar el padding constante por un padding de secuencia
    if len(row['y_preds']) > 0:
        padding = language.memory_size - len(row['y_preds'])
        padding_y_preds = torch.nn.ConstantPad1d((0, padding), 0)
        y_preds = torch.Tensor(np.array(row['y_preds'])).permute(1, 0)
        y_preds = padding_y_preds(y_preds)
    else:
        y_preds = torch.zeros((X.shape[0], language.memory_size))
    obs_data = torch.cat((X, y, y_preds), dim=1)

    # Target Expression
    padding_right = language.max_len - len(row['Target Expression'].traversal)
    padding_expr = torch.nn.ConstantPad1d((0, padding_right), language.padding_idx)
    expr = row['Target Expression'].traversal
    expr = padding_expr(torch.Tensor(expr))

    # Input Expressions
    if len(row['Input Expressions']) > 0:
        input_expressions = []
        for i in range(len(row['Input Expressions'])):
            input_expr = [language.ini_idx] + row['Input Expressions'][i].traversal
            input_expr = torch.Tensor(input_expr)
            padding_right = language.max_len + 1 - len(input_expr)
            padding_expr = torch.nn.ConstantPad1d((0, padding_right), language.padding_idx)
            input_expr = padding_expr(input_expr)
            input_expressions.append(input_expr)
        input_expressions.append(language.ini_idx*torch.ones(1))
        input_expressions = torch.cat(input_expressions, dim=0)
    else:
        input_expressions = language.ini_idx*torch.ones(1)

    return obs_data, input_expressions, expr  # Alomejor cambiarlo para que sea una namedtuple