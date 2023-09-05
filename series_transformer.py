import io
import random
from typing import List
from xmlrpc.client import Boolean
from numpy import number
import torch
from torch import nn
from torchtext.data import get_tokenizer
from torch import Tensor
import math
import configparser
from torch.utils.data import Dataset, DataLoader
import copy
from torchtext.data.metrics import bleu_score
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import R2Score
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn.functional import normalize


# custom util transformer entity
# CUTE

class Utils():
    @staticmethod
    def tokenize(text: str) -> List[str]:
        tokenizer = get_tokenizer("moses")
        result = tokenizer(text)
        result.insert(0,"<sos>")
        result.append("<eos>")
        return result

    @staticmethod
    def yield_tokens(file_path):
        with io.open(file_path, encoding = 'utf-8') as f:
            for line in f:
                yield map(str.lower, get_tokenizer("moses")(line))
    
    @staticmethod
    def encode_position(seq_len: int, dim_model: int, highValue = 1e4, device: torch.device = torch.device("cpu")) -> Tensor:
        pos = torch.arange(seq_len,dtype=torch.float,device=device).reshape(1,-1,1)
        dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1,1,-1)
        phase = pos / highValue ** (torch.div(dim, dim_model, rounding_mode='floor'))
        return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

# empty class so that pylance works
class ParameterProvider():
    pass

class ParameterProvider():
    def __init__(self, configname):
        if configname:
            self.config = configparser.ConfigParser()
            self.config.read(configname)
            self.dictionary = {
                "in_features": int(self.config['DIMENSIONS AND SIZES']['in_features']),
                "d_model": int(self.config['DIMENSIONS AND SIZES']['d_model']),
                "d_qk": int(self.config['DIMENSIONS AND SIZES']['d_qk']),
                "d_v": int(self.config['DIMENSIONS AND SIZES']['d_v']),
                "d_ff": int(self.config['DIMENSIONS AND SIZES']['d_ff']),
                "n_encoders": int(self.config['DIMENSIONS AND SIZES']['n_encoders']),
                "n_decoders": int(self.config['DIMENSIONS AND SIZES']['n_decoders']),
                "n_heads": int(self.config['DIMENSIONS AND SIZES']['n_heads']),
                "learning_rate": float(self.config['TRAINING PARAMETERS']['learning_rate']),
                "epochs": int(self.config['TRAINING PARAMETERS']['epochs']),
                "dropout": float(self.config['TRAINING PARAMETERS']['dropout'])
            }
        else:
            self.dictionary = {
                "in_features": 0,
                "d_model": 0,
                "d_qk": 0,
                "d_v": 0,
                "d_ff": 0,
                "n_encoders": 0,
                "n_decoders": 0,
                "n_heads": 0,
                "learning_rate": 1.0,
                "epochs": 0
            }
    
    def modifyWithArray(self, arr: List) -> None:
        self.dictionary = {
            "d_model": arr[0],
            "d_qk": arr[1],
            "d_v": arr[2],
            "d_ff": arr[3],
            "n_encoders": arr[4],
            "n_decoders": arr[5],
            "n_heads": arr[6],
            "epochs": self.dictionary["epochs"],
            "learning_rate": self.dictionary["learning_rate"],
            "dropout": self.dictionary["dropout"]
        }

    def getArray(self) -> List:
        return [
            self.dictionary["d_model"],
            self.dictionary["d_qk"],
            self.dictionary["d_v"],
            self.dictionary["d_ff"],
            self.dictionary["n_encoders"],
            self.dictionary["n_decoders"],
            self.dictionary["n_heads"],
            #self.dictionary["epochs"],
        ]

    def provide(self, key: str) -> any:
        return self.dictionary[key]
        
    def change(self, key: str, value: number) -> None:
        self.dictionary[key] = value

    def getChangedCopy(self,key: str, value:number) -> ParameterProvider:
        pp = copy.deepcopy(self)
        pp.change(key,value)
        return pp

class PositionalEncoding(nn.Module):

    def __init__(self, config: ParameterProvider):
        super().__init__()
        self.d_model = config.provide('d_model')
        self.n = 10000
        self.dropout = nn.Dropout(p=config.provide('dropout'))
    

    def forward(self, input_x: Tensor) -> Tensor:

        seq_len = input_x.size(1)
        pe = torch.zeros(seq_len, self.d_model)
        for k in range(seq_len):
            for i in range(int(self.d_model/2)):
                denominator = math.pow(self.n, 2*i/self.d_model)
                pe[k, 2*i] = math.sin(k/denominator)
                pe[k, 2*i+1] = math.cos(k/denominator)
        pe = torch.stack([pe for _ in range(input_x.size(0))])
        pe = pe.cuda(device=input_x.device)
        return self.dropout(torch.add(input_x,pe))
    
class InputTransformation(nn.Module):

    def __init__(self, config: ParameterProvider):
        super().__init__()
        self.linear = nn.Linear(config.provide("in_features"),config.provide("d_model"))
    
    def forward(self, input):
        return self.linear(input)

class AttentionHead(nn.Module):
    def __init__(self, config: ParameterProvider, masked: bool = False, d_v_override: int = None, d_qk_override: int = None):
        super().__init__()
        self.l = nn.Linear(100,200)
        self.masked = masked
        self.d_model = config.provide("d_model")
        self.d_qk = config.provide("d_qk") if d_qk_override is None else d_qk_override
        self.d_v = config.provide("d_v") if d_v_override is None else d_v_override
        self.WQ = nn.Linear(self.d_model,self.d_qk)
        self.WK = nn.Linear(self.d_model,self.d_qk)
        self.WV = nn.Linear(self.d_model,self.d_v)
    
    def forward(self,input_q: Tensor, input_k: Tensor, input_v: Tensor) -> Tensor:
        Q = self.WQ(input_q)
        K = self.WK(input_k)
        V = self.WV(input_v)

        if self.masked:
            if Q.size(1) != K.size(1):
                raise TypeError('Masking can be only performed when Querry and Key Matrices have the same sizes (i.e. their product is square)')
            mask = torch.stack([torch.triu(torch.full((Q.size(1),Q.size(1)),-1*torch.inf),diagonal=1) for _ in range(Q.size(0))])
            mask = mask.to(device=Q.device)
            return torch.bmm(torch.softmax(torch.add(torch.bmm(Q,torch.transpose(K,1,2)),mask)/math.sqrt(self.d_model),dim=-1),V)
        else:
            return torch.bmm(torch.softmax(torch.bmm(Q,torch.transpose(K,1,2))/math.sqrt(self.d_model),dim=-1),V)


class MultiHeadedAttention(nn.Module):
    def __init__(self,config: ParameterProvider, masked = False, d_v_override = None, d_qk_ovveride: int = None, n_heads_override = None):
        super().__init__()
        self.d_model = config.provide("d_model")
        self.d_v = config.provide("d_v") if d_v_override is None else d_v_override
        self.d_qk = config.provide("d_qk") if d_qk_ovveride is None else d_qk_ovveride
        self.n_heads = config.provide("n_heads") if n_heads_override is None else n_heads_override
        self.heads = nn.ModuleList([AttentionHead(config,masked=masked,d_qk_override=d_qk_ovveride,d_v_override=d_v_override) for _ in range(self.n_heads)])
        self.linear = nn.Linear(self.d_v*self.n_heads,self.d_model)
        self.masked = masked
    
    def forward(self, input_q, input_k, input_v):
        concatResult = torch.cat([h(input_q, input_k, input_v) for h in self.heads], dim = -1)
        return self.linear(concatResult)


class EncoderLayer(nn.Module):
    def __init__(self,config: ParameterProvider, randomize = False):
        super().__init__()
        self.d_model = config.provide("d_model")
        self.d_ff = config.provide("d_ff")
        if randomize:
            dff = self.d_ff
            ubound = int(dff * 1.1)+1
            lbound = int(dff*0.9)-1
            if lbound < 2:
                lbound = 1
            dff = random.randint(lbound,ubound)
            self.feed_forward = nn.Sequential(nn.Linear(self.d_model,dff),nn.ReLU(),nn.Linear(dff,self.d_model))
        else:
            self.feed_forward = nn.Sequential(nn.Linear(self.d_model,self.d_ff),nn.ReLU(),nn.Linear(self.d_ff,self.d_model))
        self.mha = MultiHeadedAttention(config)
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self,input_data: Tensor) -> Tensor:
        mha_output = self.mha(input_data,input_data,input_data)
        intermediate = self.norm(torch.add(mha_output,input_data))
        return self.norm(torch.add(intermediate,self.feed_forward(intermediate)))

class DecoderLayer(nn.Module):
    def __init__(self, config: ParameterProvider, randomize = False):
        super().__init__()
        self.d_model = config.provide("d_model")
        self.d_ff = config.provide("d_ff")
        if randomize:
            dff = self.d_ff
            ubound = int(dff * 1.1)+1
            lbound = int(dff*0.9)-1
            if lbound < 2:
                lbound = 1
            dff = random.randint(lbound,ubound)
            self.feed_forward = nn.Sequential(nn.Linear(self.d_model,dff),nn.ReLU(),nn.Linear(dff,self.d_model))
        else:
            self.feed_forward = nn.Sequential(nn.Linear(self.d_model,self.d_ff),nn.ReLU(),nn.Linear(self.d_ff,self.d_model))
        self.self_mha = MultiHeadedAttention(config, masked = True)
        self.ed_mha = MultiHeadedAttention(config)
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self,input_data: Tensor, encoder_data: Tensor) -> Tensor:
        mha_output = self.self_mha(input_data,input_data,input_data)
        intermediate = self.norm(torch.add(mha_output,input_data))
        mha_output = self.ed_mha(input_data,encoder_data,encoder_data)
        intermediate = self.norm(torch.add(mha_output,intermediate))
        return self.norm(torch.add(intermediate,self.feed_forward(intermediate)))

class EncoderStack(nn.Module):
    def __init__(self, config: ParameterProvider):
        super().__init__()
        self.n_encoders = config.provide("n_encoders")
        self.encoders = nn.ModuleList([EncoderLayer(config) for _ in range(self.n_encoders)])
    
    def forward(self, input_data: Tensor) -> Tensor:
        for l in self.encoders:
            input_data = l(input_data)
        return input_data

class NumericalOut(nn.Module):
    def __init__(self, config: ParameterProvider):
        super().__init__()
        self.linear = nn.Linear(config.provide("d_model"),config.provide("in_features"))
    
    def forward(self, input_data: Tensor) -> Tensor:
        #return torch.softmax(self.linear(input_data), dim = -1)
        return self.linear(input_data)


class DecoderStack(nn.Module):
    def __init__(self, config: ParameterProvider):
        super().__init__()
        self.n_decoders = config.provide("n_decoders")
        self.decoders = nn.ModuleList([DecoderLayer(config) for _ in range(self.n_decoders)])
    
    def forward(self, input_data: Tensor, ed_data: Tensor) -> Tensor:
        for l in self.decoders:
            input_data = l(input_data,ed_data)
        return input_data

class Transformer(nn.Module):
    def __init__(self, config: ParameterProvider, mask = True):
        super().__init__()
        self.config = config
        self.d_model = config.provide("d_model")
        self.encoder_input_transformation = InputTransformation(config)
        self.decoder_input_transformation = InputTransformation(config)
        self.pos_encoding_in = PositionalEncoding(config)
        self.pos_encoding_out = PositionalEncoding(config)
        self.encoder_stack = EncoderStack(config)
        self.decoder_stack = DecoderStack(config)
        self.out = NumericalOut(config)

    def setMasking(self, mask: Boolean):
        for d in self.decoder_stack.decoders:
            for a in d.ed_mha.heads:
                a.masked = mask


    def forward(self, encoder_input: Tensor, decoder_input: Tensor):

        encoder_input = self.encoder_input_transformation(encoder_input)
        encoder_input = self.pos_encoding_in(encoder_input)
        encoder_out = self.encoder_stack(encoder_input)

        decoder_input = self.decoder_input_transformation(decoder_input)
        decoder_input = self.pos_encoding_out(decoder_input)
        decoder_out = self.decoder_stack(decoder_input,encoder_out)

        output = self.out(decoder_out)
        return output
        #return self.lexical_out(numerical)


class CustomDataSet(Dataset):
    def __init__(self, infile: str, window_length = 32, prediction_window = 7, require_date_split = True, drop_idx_column = False):

        file = open(infile,'r')
        df = pd.read_csv(file)
        if require_date_split:
            df[["day", "month", "year"]] = df["Date"].str.split("/", expand = True)
            df['day'] = df['day'].astype(float)
            df['month'] = df['month'].astype(float)
            df['year'] = df['year'].astype(float)
            df = df.drop(['Date'],axis=1)
        if drop_idx_column:
            df = df.drop(['ID'],axis=1)
        data = torch.tensor(df.values)

        print(list(df.columns))

        # normalize (is this correct way to normalize this?)

        for i in range(data.size(1)):
            epsilon = torch.zeros_like(data[:,i])
            epsilon.fill_(1e-4)
            data[:,i] = (data[:,i] - torch.min(data[:,i])) / torch.max(epsilon,(torch.max(data[:,i]) - torch.min(data[:,i])))

        #data = torch.nn.functional.normalize(data, dim=0)

        source_samples = []
        shifted_target = []
        target_samples = []

        for i in range(window_length, data.size()[0]-prediction_window):
            source_samples.append(data[i-window_length:i,:])
            shifted_target.append(data[i-1:i+prediction_window-1,:])
            target_samples.append(data[i:i+prediction_window,:])

        self.L = len(range(window_length, data.size()[0]-prediction_window))

        self.Xe = source_samples
        self.Xd = shifted_target
        self.Y = target_samples
        '''
                self.Xe = torch.nn.utils.rnn.pad_sequence(self.Xe,batch_first=True)
                self.Xd = torch.nn.utils.rnn.pad_sequence(self.Xd,batch_first=True)
                self.Y = torch.nn.utils.rnn.pad_sequence(self.Y,batch_first=True)
        '''
    def __len__(self):
        return self.L

    def __getitem__(self, index):
        _xe = self.Xe[index]
        _xd = self.Xd[index]
        _y = self.Y[index]
        return _xe.to(torch.float), _xd.to(torch.float), _y.to(torch.float)

    def getSets(self, split = 0.8):
        train_size = int(0.8 * len(self))
        test_size = len(self) - train_size

        # Created using indices from 0 to train_size.
        train_dataset = torch.utils.data.Subset(self, range(train_size))

        # Created using indices from train_size to train_size + test_size.
        test_dataset = torch.utils.data.Subset(self, range(train_size, train_size + test_size))

        return train_dataset,test_dataset



def train_cuda(model: Transformer, train_dataset: CustomDataSet, device: int, batch_size = 32, lr: float = 0.001, epochs: int = 1, verbose_delay = 100) -> None:
    
    model.cuda(device=device)
    criterion = nn.MSELoss().cuda(device=device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    epoch_loss_history = []

    epoch_loss = 0.
    for epoch in range(epochs):

        print('Epoch ' + str(epoch)+' of '+str(epochs))
        epoch_loss = 0.
        for i, (xe, xd, y) in enumerate(data_loader):
            if verbose_delay > 0 and i%verbose_delay== 0:
                print(str(i) + " of " + str(len(data_loader)))

            last_loss = 0.
            xe = xe.cuda(device)
            xd = xd.cuda(device)

            y = y.cuda(device)

            optimizer.zero_grad()
            output = model(xe, xd)

            loss = criterion(output,y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            last_loss = loss.item()
            epoch_loss += float(last_loss)
            del loss
            del output
        epoch_loss /= len(data_loader)

        epoch_loss_history.append(epoch_loss)

        print("Epoch loss: "+str(epoch_loss))
        


    return epoch_loss, epoch_loss_history

def validate_cuda(model: Transformer, test_dataset: CustomDataSet, device: int, batch_size = 32, verbose_delay = 100):
    model.cuda(device=device)
    criterion = nn.MSELoss().cuda(device=device)
    data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    final_mse = 0.
    with torch.no_grad:
        for i, (xe, xd, y) in enumerate(data_loader):
            final_mse += criterion(model(xe,xd),y)
    
    return  final_mse / len(data_loader)

def output_and_show(model: Transformer, input: Tensor, output: Tensor, device: int, visualize_index: int = 0):
    decoder_input = torch.cat((torch.unsqueeze(input[0],dim=0),output[0:-1]))
    
    model.cuda(device=device)
    predicted_output = model(torch.unsqueeze(input,dim=0).to(device),torch.unsqueeze(decoder_input,dim=0).to(device)).cpu()
    input_list = input[:,visualize_index].tolist()
    output_list = output.cpu()[:,visualize_index].tolist()
    predicted_list = predicted_output[0].detach()[:,visualize_index].tolist()
    numbers_pre_prediction = list(range(len(input_list)))
    numbers_post_prediction = list(range(len(input_list),len(input_list)+len(output_list)))
    print("R^2", R2Score()(torch.tensor(output_list),torch.tensor(predicted_list)), flush = True)
    plt.plot(numbers_pre_prediction, input_list,'k-')
    plt.plot(numbers_post_prediction, output_list,'b-')
    plt.plot(numbers_post_prediction, predicted_list,'r-')
    plt.show()


def long_range_output_and_show(model: Transformer, input: Tensor, output: Tensor, device: int, visualize_index: int = 0):
    decoder_input = torch.cat((torch.unsqueeze(input[0],dim=0),output[0:-1]))
    
    model.cuda(device=device)
    predicted_output = model(torch.unsqueeze(input,dim=0).to(device),torch.unsqueeze(decoder_input,dim=0).to(device)).cpu()
    print(output.size())
    print(predicted_output.size())
    plt.plot(output.cpu()[:,visualize_index],'b-')
    plt.plot(predicted_output.detach()[0,:,visualize_index],'r-')
    plt.show()