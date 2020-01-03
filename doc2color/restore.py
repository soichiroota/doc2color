from models import Net, BertModelWithTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np

import argparse

class Doc2Color:
    def __init__(self, doc2vec, vec2color):
        self.doc2vec = doc2vec
        self.vec2color = vec2color

    def get_color_from_doc(self, doc):
        vec = self.doc2vec.get_sentence_embedding(doc)
        tensor = torch.tensor(vec).reshape(1, -1)
        color = self.vec2color(tensor)
        return color[0].detach().numpy()


def get_color_code(color):
    rgb = np.round(color * 255.0).astype(int)
    return '#%02x%02x%02x' % tuple(rgb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Vec2Color Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    bert_model = BertModelWithTokenizer("bert-base-multilingual-cased")

    model = Net().to(device)
    model.load_state_dict(torch.load("doc2color/pt_objects/vec2color.pt"))
    model.eval()

    doc2color = Doc2Color(bert_model, model)
    for color_name in ('red', 'レッド', '赤', 'あか', 'green', 'グリーン', '緑', 'みどり', 'blue', 'ブルー', '青',  'あお', 'black', 'ブラック', '黒', 'くろ', 'white', 'ホワイト', '白', 'しろ', 'gray', 'グレー', '灰色', 'はいいろ'):
        names = (
            color_name.capitalize(),
            color_name,
            color_name.upper(),
            color_name.title())
        for name in names:
            color = doc2color.get_color_from_doc(name)
            color_code = get_color_code(color)
            print(name, color_code)