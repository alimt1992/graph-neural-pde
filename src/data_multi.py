import os
import argparse
import json
from turtle import forward
import h5py
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import io
from skimage.transform import resize
from graph_rewiring import make_symmetric, apply_pos_dist_rewire,\
    get_two_hop, apply_gdc, to_undirected


def rewire(data, opt, data_dir):
  rw = opt['rewiring']
  if rw == 'two_hop':
    data = get_two_hop(data)
  elif rw == 'gdc':
    data = apply_gdc(data, opt)
  elif rw == 'pos_enc_knn':
    data = apply_pos_dist_rewire(data, opt, data_dir)
  return data

def get_dataset(opt: dict, data_dir, device='cpu') -> Dataset:
  ds = opt['dataset']
  path = os.path.join(data_dir, ds)
  if ds == 'CLEVR_v1':
    train_dataset = CLEVR_v1('CLEVR_train_questions.json', data_dir, split='train', region_propssals=False,
                             graph_size=9, batch_size=opt['batch_size'], device=device)
    val_dataset = CLEVR_v1('CLEVR_val_questions.json', data_dir, split='val', region_propssals=False,
                           graph_size=9, batch_size=opt['batch_size'], device=device)
  else:
    raise Exception('Unknown dataset.')
  
#   if opt['rewiring'] is not None:
#     train_dataset.data = rewire(train_dataset.data, opt, data_dir)
  
  return train_dataset, val_dataset


def get_multimodal_opt(opt):
#   opt['im_dataset'] = 'MNIST'  #datasets = ['MNIST','CIFAR']

  opt['input_dropout'] = 0.5
  opt['dropout'] = 0
  opt['optimizer'] = 'rmsprop'
  opt['lr'] = 0.0047
  opt['decay'] = 5e-4
  opt['self_loop_weight'] = 0.555  #### 0?
  opt['alpha'] = 0.918
  opt['time'] = 12.1
  opt['augment'] = False #True   #False need to view image
  opt['attention_dropout'] = 0
  opt['adjoint'] = False

  opt['epoch'] = 4 #3 #10#20 #400
  opt['batch_size'] = 64 #64  # doing batch size for mnist
  opt['train_size'] = 512 #128#512 #4096 #2048 #512 #2047:#4095:#511:#5119:#1279:#
  opt['test_size'] = 64 #2559:#1279:
  assert (opt['train_size']) % opt['batch_size'] == 0, "train_size needs to be multiple of batch_size"
  assert (opt['test_size']) % opt['batch_size'] == 0, "test_size needs to be multiple of batch_size"


  if opt['dataset'] == 'MNIST':
    opt['im_width'] = 28
    opt['im_height'] = 28
    opt['im_chan'] = 1
    opt['hidden_dim'] = 1 #16    #### 1 or 3 rgb?
    opt['num_feature'] = 1  # 1433   #### 1 or 3 rgb?
    opt['num_class'] = 10  # 7  #### mnist digits

  elif opt['dataset'] == 'CIFAR':
    opt['im_width'] = 32
    opt['im_height'] = 32
    opt['im_chan'] = 3
    # ????
    opt['hidden_dim'] = 3 #16    #### 1 or 3 rgb?
    opt['num_feature'] = 3  # 1433   #### 1 or 3 rgb?
    opt['num_class'] = 10  # 7  #### mnist digits

  elif opt['dataset'] == 'CLEVR':
    opt['im_width'] = 224
    opt['im_height'] = 224
    opt['im_chan'] = 3
    # ????
    opt['hidden_dim'] = [1024, 384] #16    #### 1 or 3 rgb?
    opt['num_feature'] = [2048, 768]  # 1433   #### 1 or 3 rgb?
    opt['num_class'] = 10  # 7  #### mnist digits

  opt['num_nodes'] = 9
  opt['simple'] = False #True
  opt['diags'] = True
  opt['ode'] = 'ode' #'att' don't think att is implmented properly on this codebase?
  opt['linear_attention'] = True
  opt['batched'] = True
  return opt

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def object_edge_index_calc():
    pass

def grid_edge_index_calc(grid_height, grid_width, diags = False):
    edge_list = []
    def oneD():
        for i in range(grid_height * grid_width):
            #corners
            if i in [0, grid_width-1, grid_height * grid_width - grid_width, grid_height * grid_width - 1]:
                if i  == 0:
                    edge_list.append([i,1])
                    edge_list.append([i,grid_width])
                    edge_list.append([i,grid_width + 1]) if diags == True else 0
                elif i == grid_width - 1:
                    edge_list.append([i, grid_width - 2])
                    edge_list.append([i, 2 * grid_width - 1])
                    edge_list.append([i, 2 * grid_width - 2]) if diags == True else 0
                elif i == grid_height * grid_width - grid_width:
                    edge_list.append([i, grid_height * grid_width - 2*grid_width])
                    edge_list.append([i, grid_height * grid_width - grid_width+1])
                    edge_list.append([i, grid_height * grid_width - 2*grid_width+1]) if diags == True else 0
                elif i == grid_height * grid_width - 1:
                    edge_list.append([i, grid_height * grid_width - 2])
                    edge_list.append([i, grid_height * grid_width - 1 - grid_width])
                    edge_list.append([i, grid_height * grid_width - grid_width - 2]) if diags == True else 0
            # top edge
            elif i in range(1,grid_width-1):
                edge_list.append([i,i-1])
                edge_list.append([i,i+1])
                edge_list.append([i,i+grid_width])
                if diags:
                    edge_list.append([i, i + grid_width -1])
                    edge_list.append([i, i + grid_width + 1])
            # bottom edge
            elif i in range(grid_height * grid_width - grid_width, grid_height * grid_width):
                edge_list.append([i,i-1])
                edge_list.append([i,i+1])
                edge_list.append([i,i-grid_width])
                if diags:
                    edge_list.append([i, i - grid_width -1])
                    edge_list.append([i, i - grid_width + 1])
            # middle
            else:
                if i % grid_width == 0: # left edge
                    edge_list.append([i,i+1])
                    edge_list.append([i,i-grid_width])
                    edge_list.append([i,i+grid_width])
                    if diags:
                        edge_list.append([i, i - grid_width + 1])
                        edge_list.append([i, i + grid_width + 1])
                elif (i + 1) % grid_width == 0: # right edge
                    edge_list.append([i,i-1])
                    edge_list.append([i,i-grid_width])
                    edge_list.append([i,i+grid_width])
                    if diags:
                        edge_list.append([i, i - grid_width - 1])
                        edge_list.append([i, i + grid_width - 1])
                else:
                    edge_list.append([i,i-1])
                    edge_list.append([i,i+1])
                    edge_list.append([i,i-grid_width])
                    edge_list.append([i,i+grid_width])
                    if diags:
                        edge_list.append([i, i - grid_width - 1])
                        edge_list.append([i, i - grid_width + 1])
                        edge_list.append([i, i + grid_width - 1])
                        edge_list.append([i, i + grid_width + 1])
        return edge_list

    edge_list = oneD()
    ret_edge_tensor = torch.tensor(edge_list).T
    if diags:
        assert ret_edge_tensor.shape[1] == (8*(grid_width-2)*(grid_height-2)\
                                    + 2*5*(grid_width-2) + 2*5*(grid_height-2)\
                                    + 4*3) ,"Wrong number of fixed grid edges (inc diags)"
    else:
        assert ret_edge_tensor.shape[1] == (4*(grid_width-2)*(grid_height-2) \
                                    + 2*3*(grid_width-2) + 2*3*(grid_height-2)\
                                    + 4*2) ,"Wrong number of fixed grid edges (exc diags)"
    return ret_edge_tensor


def text_edge_index_calc(num_tokens):
    edge_list = []
    for i in range(num_tokens - 1):
        edge_list.append([i, i + 1])
    ret_edge_tensor = torch.tensor(edge_list).T
    return ret_edge_tensor



class ImageTransform(torch.nn.Module):
    def __init__(self, region_propssals=False, graph_size=9, batch_size=16, model='resnet101', rp_model=None):
        super(ImageTransform, self).__init__()
        self.region_propssals = region_propssals
        self.graph_size = graph_size
        self.batch_size = batch_size

        if not hasattr(torchvision.models, model):
            raise ValueError('Invalid model "%s"' % args.model)
        cnn = getattr(torchvision.models, model)(pretrained=True)
        self.model = torch.nn.Sequential(*list(cnn.children())[:-1])
        self.model.add_module('flatten', torch.nn.Flatten())
        self.model.to(self.device)
        self.model.eval()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        
        if region_propssals:
            raise NotImplementedError("Region Proposals is not implemented yet!")
        
        
        grid_height = np.ceil(np.sqrt(graph_size), dtype='int32')
        grid_width = np.ceil(np.sqrt(graph_size), dtype='int32')
        i_graph = grid_edge_index_calc(grid_height, grid_width)
        self.i_graph = i_graph.unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
    
    def forward(self, image_list):
        image_tensor = []
        i_graph = self.i_graph
        for i in len(image_list):
            img = io.imread(image_list[i])

            if self.region_propssals:
                raise NotImplementedError("Region Proposals is not implemented yet!")
            else:
                grid_height = np.ceil(np.sqrt(self.graph_size), dtype='int32')
                grid_width = np.ceil(np.sqrt(self.graph_size), dtype='int32')
                img = resize(img, (grid_height * 224, grid_width * 224),
                           anti_aliasing=True).transpose((2, 0, 1))
                img = img.reshape(img.shape[0], grid_height, 224, grid_width, 224).transpose(1, 3, 0, 2, 4)
                img = img. reshape(-1, img.shape[2], img.shape[3], img.shape[4]) / 255.
                with torch.no_grad():
                    img = self.model(self.normalize(torch.from_numpy(img)))
                    image_tensor.append(img)
        
        image_tensor = torch.stack(image_tensor, dim=0)

        return image_tensor, i_graph

class TextTransform(torch.nn.Module):
    def __init__(self, batch_size=16, model='bert-base-cased'):
        super(TextTransform, self).__init__()
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', model)
        self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', model)
        self.batch_size = batch_size
        self.max_tokens = 0
        self.q_graph = None

    def get_max_tokens(self, data):
        max_tokens = 0
        tokens = self.tokenizer.encode([data[0]['question']], padding='max_length')
        max_tokens = tokens.size(1)
        self.max_tokens = max_tokens

        q_graph = grid_edge_index_calc(max_tokens)
        self.q_graph = q_graph.unsqueeze(0).expand(self.batch_size, -1, -1).to(self.device)

        return max_tokens

    def forward(self, data):
        questions = []
        image_indices = []

        for i in len(data):
            questions.append(data[i]['question'])
            image_indices.append(data[i]['image_index'])
        
        questions = self.tokenizer.encode(questions, padding='max_length', truncation=True)
        questions = torch.tensor(questions)
        image_indices = torch.tensor(image_indices)
        with torch.no_grad():
            questions, _ = self.model(questions)

        return questions, self.q_graph, image_indices

class LabelTransform(torch.nn.Module):
    def __init__(self, batch_size=16):
        super(LabelTransform, self).__init__()
        self.batch_size = batch_size
        self.answer_list = []

    def get_answer_list(self, data):
        answer_list = []
        for i in len(data):
            answer_list.append(data[i]['answer'])
        
        answer_list = np.array(answer_list)
        answer_list = np.unique(answer_list)
        self.answer_list = answer_list

        return answer_list

    def forward(self, data):
        answers = []

        for i in len(data):
            answers.append(np.where(self.answer_list == data[i]['answer']))
        
        answers = np.array(answers).squeeze()

        return answers


class MultiModalGraphDataset(Dataset):
    def __init__(self, annotations_file, data_dir, modality_transforms=None, target_transform=None, num_modalities=None, device=torch.device('cpu')):
        self.labels = pd.read_csv(annotations_file)
        self.data_dir = data_dir
        self.num_modalities = num_modalities
        self.modality_transforms = modality_transforms
        self.target_transform = target_transform
        self.device = device

    def __len__(self):
        return len(self.labels)

class CLEVR_v1(MultiModalGraphDataset):
    def __init__(self, annotations_file, data_dir, split='train', region_propssals=False, graph_size=9, batch_size=16, device=torch.device('cpu')):
        super(CLEVR_v1, self).__init__(annotations_file, data_dir, None, None, 2, device)

        self.split = split
        rp = '_rp' if region_propssals else '_norp'.format(graph_size)

        if not os.path.exists(data_dir + '/data/CLEVR_v1.0_' + split + rp  + '.h5'):
            if not os.path.exists(data_dir + '/data/CLEVR_v1.0.zip' + split + '.h5'):
                cmd = 'mkdir  ' + data_dir + '/data'
                os.system(cmd)
                cmd = 'wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip -O ' + data_dir + '/data/CLEVR_v1.0.zip'
                os.system(cmd)
                cmd = 'unzip  ' + data_dir + '/data/CLEVR_v1.0.zip -d ' + data_dir + '/data'
                os.system(cmd)
            self.data_dir = data_dir + '/data/CLEVR_v1.0'
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            image_transform = ImageTransform(region_propssals=False, graph_size=9, batch_size=batch_size).to(device)
            text_transform = TextTransform(batch_size=batch_size).to(device)
            self.modality_transforms = [image_transform, text_transform]
            self.target_transform = LabelTransform(batch_size=batch_size).to(device)

            with h5py.File(data_dir + '/data/CLEVR_v1.0_' + split + rp + '.h5','w') as f:
                DIR = data_dir + '/data/CLEVR_v1.0/images/' + split
                images_list = [os.path.join(DIR, name) for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
                num_images = len(images_list)
                max_i_edges = np.ceil(np.sqrt(graph_size), dtype='int32')
                max_i_edges = 2 * max_i_edges * (max_i_edges - 1)

                f.create_dataset('images', (num_images, graph_size, 2048), chunks=True, dtype='float64')
                f.create_dataset('i_graph', (num_images, 2, max_i_edges), chunks=True, dtype='int32')

                for i in range(np.ceil(num_images / batch_size, dtype='int32')):
                    start = i * batch_size
                    end = (i + 1) * batch_size if ((i + 1) * batch_size) < num_images else num_images
                    im, gr = self.modality_transforms[0](images_list[start: end])
                    f['images'][start: end] = im.numpy()
                    f['i_graph'][start: end] = gr.numpy()
                
                DIR = data_dir + '/data/CLEVR_v1.0/questions/' + annotations_file
                with open(DIR) as j:
                    data = json.load(j)
                    num_questions = len(data['questions'])
                    max_tokens = self.modality_transforms[1].get_max_tokens(data['questions'])
                    if split == 'train':
                        answer_list = self.target_transform.get_answer_list(data['questions'])

                    f.create_dataset('questions', (num_questions, max_tokens, 768), chunks=True, dtype='float64')
                    f.create_dataset('q_graph', (num_questions, 2, max_tokens - 1), chunks=True, dtype='int32')
                    f.create_dataset('image_indices', (num_questions), chunks=True, dtype='int32')
                    if split != 'test':
                        f.create_dataset('answers', (num_questions), chunks=True, dtype='int32')
                    if split == 'train':
                        f.create_dataset('answers_list', (len(answer_list)), chunks=True, dtype='int32')
                        f['answers_list'] = answer_list

                    for i in range(np.ceil(num_questions / batch_size, dtype='int32')):
                        start = i * batch_size
                        end = (i + 1) * batch_size if ((i + 1) * batch_size) < num_questions else num_questions
                        q, gr, idx = self.modality_transforms[1](data['questions'][start: end])
                        f['questions'][start: end] = q.numpy()
                        f['q_graph'][start: end] = gr.numpy()
                        f['image_indices'][start: end] = idx.numpy()
                        if split != 'test':
                            ans = self.target_transform(data['answers'][start: end])
                            f['answers'][start: end] = ans.numpy()            
        
        self.h5py_file = h5py.File(data_dir + '/data/CLEVR_v1.0_' + split + rp + '.h5', 'r')
        self.num_node_features = [2048, 768]
        if split == 'train':
            self.num_classes = len(answer_list)
            

    def __getitem__(self, idx):
        image_index = self.h5py_file['image_indices'][idx]
        image = self.h5py_file['images'][image_index]
        question = self.h5py_file['questions'][idx]
        i_graph = self.h5py_file['i_graph'][image_index]
        q_graph = self.h5py_file['q_graph'][idx]
        answer = self.h5py_file['answers'][idx] if self.split != 'test' else None

        image_index = torch.from_numpy(image_index).to(self.device)
        image = torch.from_numpy(image).to(self.device)
        question = torch.from_numpy(question).to(self.device)
        i_graph = torch.from_numpy(i_graph).to(self.device)
        q_graph = torch.from_numpy(q_graph).to(self.device)
        answer = torch.from_numpy(answer).to(self.device) if self.split != 'test' else None

        sample = {'modality_data': [image, question],
                  'modility_graphs': [i_graph, q_graph],
                  'additional_data': None,
                  'labels': answer}

        return sample
    
    def get_answer_list(self):
        return self.h5py_file['answers_list']
    
    def __len__(self):
        return len(self.h5py_file['questions'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_image_defaults', default='MNIST',
                        help='#Image version# Whether to run with best params for cora. Overrides the choice of dataset')
    # parser.add_argument('--use_image_defaults', action='store_true',
    #                     help='Whether to run with best params for cora. Overrides the choice of dataset')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')  ######## NEED
    parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
    parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
    parser.add_argument('--epoch', type=int, default=10, help='Number of training epochs per iteration.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
    parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
    parser.add_argument('--augment', action='store_true',
                        help='double the length of the feature vector by appending zeros to stabilist ODE learning')
    parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
    parser.add_argument('--alpha_sigmoid', type=bool, default=True, help='apply sigmoid before multiplying by alpha')
    parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
    # ODE args
    parser.add_argument('--method', type=str, default='dopri5',
                        help="set the numerical solver: dopri5, euler, rk4, midpoint")  ######## NEED
    parser.add_argument('--ode', type=str, default='ode',
                        help="set ode block. Either 'ode', 'att', 'sde'")  ######## NEED
    parser.add_argument('--adjoint', default=False, help='use the adjoint ODE method to reduce memory footprint')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
    parser.add_argument('--simple', type=bool, default=False,
                        help='If try get rid of alpha param and the beta*x0 source term')
    # SDE args
    parser.add_argument('--dt_min', type=float, default=1e-5, help='minimum timestep for the SDE solver')
    parser.add_argument('--dt', type=float, default=1e-3, help='fixed step size')
    parser.add_argument('--adaptive', type=bool, default=False, help='use adaptive step sizes')
    # Attention args
    parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                        help='slope of the negative part of the leaky relu used in attention')
    parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
    parser.add_argument('--heads', type=int, default=1, help='number of attention heads')
    parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
    parser.add_argument('--attention_dim', type=int, default=64,
                        help='the size to project x to before calculating att scores')
    parser.add_argument('--mix_features', type=bool, default=False,
                        help='apply a feature transformation xW to the ODE')
    parser.add_argument('--linear_attention', type=bool, default=False,
                        help='learn the adjacency using attention at the start of each epoch, but do not update inside the ode')
    parser.add_argument('--mixed_block', type=bool, default=False,
                        help='learn the adjacency using a mix of attention and the Laplacian at the start of each epoch, but do not update inside the ode')

    # visualisation args
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--batched', type=bool, default=True,
                        help='Batching')
    parser.add_argument('--im_width', type=int, default=28, help='im_width')
    parser.add_argument('--im_height', type=int, default=28, help='im_height')
    parser.add_argument('--diags', type=bool, default=False,
                        help='Edge index include diagonal diffusion')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='MNIST, CIFAR, CLEVR')
    args = parser.parse_args()
    opt = vars(args)
    opt = get_multimodal_opt(opt)

    # load_Superpix75Mat(opt)


    # Cora = get_dataset('Cora', '../data', False)
    # gnn = GNN(self.opt, dataset, device=self.device)
    # odeblock = gnn.odeblock
    # func = odeblock.odefunc

    # img_size = 32#28
    # im_width = img_size
    # im_height = img_size
    # im_chan = 3 #1
    # exdataset = 'CIFAR' #'MNIST'

    # train_loader = torch.utils.data.DataLoader(
    #   torchvision.datasets.MNIST('data/' + exdataset + '/', train=True, download=True,
    #                              transform=torchvision.transforms.Compose([
    #                                torchvision.transforms.ToTensor(),
    #                                torchvision.transforms.Normalize(
    #                                  (0.1307,), (0.3081,))
    #                              ])),
    #                                 batch_size=1, shuffle=True)

    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #
    # train_loader = torch.utils.data.DataLoader(
    #   torchvision.datasets.CIFAR10('data/' + exdataset + '/', train=True, download=True,
    #                              transform=transform),
    #                                 batch_size=1, shuffle=True)
    #
    #
    # edge_index = edge_index_calc(im_height, im_width)
    #
    # opt = get_multimodal_opt({})
    # Graph = create_in_memory_dataset(opt, "Train", train_loader, edge_index, im_height, im_width, im_chan,
    #                                                                         root='./data/Graph' + exdataset + 'GNN/',
    #                                                                         processed_file_name='Graph' + exdataset + 'GNN2.pt')
    #
    # fig = plt.figure(figsize=(32,62))
    # # for i in range(6):
    # #     plt.subplot(2, 3, i + 1)
    #
    # for i in range(20):
    #     plt.subplot(5, 4, i + 1)
    #     plt.tight_layout()
    #     digit = Graph[i]
    #     plt.title("Ground Truth: {}".format(digit.y.item()))
    #     plt.xticks([])
    #     plt.yticks([])
    #     A = digit.x#.view(im_height, im_width, im_chan)
    #     A = A.numpy()
    #     A = np.reshape(A, (im_height, im_width, im_chan), order='F')
    #     A = A / 2 + 0.5  # unnormalize
    #     plt.imshow(np.transpose(A, (1, 0, 2)))
    # # plt.show()
    # plt.savefig("GraphImages.png", format="PNG")