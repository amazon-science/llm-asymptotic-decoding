import torch

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, w_ind_gpt2_tensor, bptt, device):
        self.w_ind_gpt2 = w_ind_gpt2_tensor
        self.seq_len = bptt
        self.output_device = device

    def __len__(self):
        return int( self.w_ind_gpt2.size(0) /self.seq_len )

    def __getitem__(self, idx):
        feature = self.w_ind_gpt2[idx*self.seq_len:(idx+1)*self.seq_len].to(dtype = torch.long, device = self.output_device)
        return feature

def create_data_loader(f_in, bsz, bptt, device, dataset_class, shuffle = True):
    w_ind_gpt2_tensor = torch.load(f_in, map_location='cpu')
    cut_tok_num = w_ind_gpt2_tensor.size(0) % bptt
    if cut_tok_num > 0:
        w_ind_gpt2_tensor = w_ind_gpt2_tensor[:-cut_tok_num]
    dataset = dataset_class(w_ind_gpt2_tensor, bptt, device)
    use_cuda = False
    if device.type == 'cuda':
        use_cuda = True
    return torch.utils.data.DataLoader(dataset, batch_size = bsz, shuffle = shuffle, pin_memory=not use_cuda, drop_last=False)
    #return torch.utils.data.DataLoader(dataset, batch_size = bsz, shuffle = shuffle, pin_memory=not use_cuda, drop_last=True)


def load_corpus(data_path, train_bsz, eval_bsz, bptt, device, tensor_folder = "tensors_all", skip_training = False, shuffle_train=True, shuffle_val = False, load_val = True, load_testing = True):
    train_corpus_name = data_path + "/" + tensor_folder + "/train.pt"
    val_org_corpus_name = data_path +"/" + tensor_folder + "/val_org.pt"
    test_org_corpus_name = data_path +"/" + tensor_folder + "/test_org.pt"

    dataloader_train = []
    dataloader_val = []
    dataloader_test = []
    
    dataset_class = SeqDataset

    if load_val:
        with open(val_org_corpus_name,'rb') as f_in:
            dataloader_val = create_data_loader(f_in, eval_bsz, bptt, device, dataset_class, shuffle = shuffle_val)

    if load_testing:
        with open(test_org_corpus_name,'rb') as f_in:
            dataloader_test = create_data_loader(f_in, eval_bsz, bptt, device, dataset_class, shuffle = shuffle_val)

    if not skip_training:
        with open(train_corpus_name,'rb') as f_in:
            dataloader_train = create_data_loader(f_in, train_bsz, bptt, device, dataset_class, shuffle = shuffle_train)

    return dataloader_train, dataloader_val, dataloader_test
