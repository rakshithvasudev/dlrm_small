import functools
from typing import Sequence
import time

import numpy as np 
import torch 
import torch.nn as nn
import argparse
from random_ds import make_random_data_and_loader
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter


def collate_wrapper_random_offset(list_of_tuples):
    # where each tuple is (X, lS_o, lS_i, T)
    (X, lS_o, lS_i, T) = list_of_tuples[0]
    return (X,
            torch.stack(lS_o),
            lS_i,
            T)


def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value


def dash_separated_floats(value):
    vals = value.split("-")
    for val in vals:
        try:
            float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of floats" % value
            )

    return value

class DlrmSmall(nn.Module):

    """Define a DLRM-Small model.
    
    Parameters:
      vocab_sizes: list of vocab sizes of embedding tables.
      total_vocab_sizes: sum of embedding table sizes (for jit compilation).
      mlp_bottom_dims: dimensions of dense layers of the bottom mlp.
      mlp_top_dims: dimensions of dense layers of the top mlp.
      num_dense_features: number of dense features as the bottom mlp input.
      embed_dim: embedding dimension.
      keep_diags: whether to keep the diagonal terms in x @ x.T.
    """


    vocab_sizes: Sequence[int]
    total_vocab_sizes: int
    num_dense_features: int
    mlp_bottom_dims: Sequence[int] = (512, 256, 128)
    mlp_top_dims: Sequence[int] = (1024, 1024, 512, 256, 1)


    def create_mlp(self, ln):
        """
        ln: layers as dims
        """

        layers = nn.ModuleList()
    
        for i in range(0, ln.size -1):

            n = ln[i]
            m = ln[i + 1]
    
            LL = nn.Linear(int(n), int(m), bias = True)
            
            # glorot uniform
            nn.init.xavier_normal_(LL.weight.data)
            # bias_init=jnn.initializers.normal(stddev=jnp.sqrt(1.0 / dense_dim))
            #nn.init.kaiming_normal_(LL.bias.data, mode='fan_out', nonlinearity='relu')
            std_dev = np.sqrt(1/m)
            bt = np.random.normal(0.0, std_dev, size=m).astype(np.float32)
            LL.bias.data = torch.tensor(bt, requires_grad=True)

            layers.append(LL)

            layers.append(nn.ReLU())

        print(torch.nn.Sequential(*layers))

        return torch.nn.Sequential(*layers)
        

    
    def create_emb(self, m, ln):
        """
        m: embedding dimensions, feature size?
        ln: arch embedding size - np array of vocab size
        """



        #print(ln)
        #print(ln.size)
        #print(type(ln))
        emb_l = nn.ModuleList()

         
        for i in range(0, ln.size):

            # vocab size of the embedding layer
            n = ln[i]

            EE = nn.EmbeddingBag(n, m, mode='sum', sparse=True)
            W = np.random.uniform(low = -np.sqrt(1/n), high=np.sqrt(1/n), size=(n, m)).astype(np.float32)

            EE.weight.data = torch.tensor(W, requires_grad=True)

            emb_l.append(EE)

        return emb_l


    def __init__(self, 
            m_spa = None,
            ln_emb = None,
            ln_bot = None,
            ln_top = None, 
            ndevices=-1):

        super(DlrmSmall, self).__init__()
        
        self.emb_l = self.create_emb(m_spa, ln_emb)
        self.bot_l = self.create_mlp(ln_bot)
        self.top_l = self.create_mlp(ln_top)


    def apply_mlp(self, x, layers):
        return layers(x)


    def apply_emb(self, lS_o, lS_i, emb_l):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        ly = []

        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]
            

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            E = emb_l[k]
            V = E(sparse_index_group_batch, sparse_offset_group_batch)

            ly.append(V)

        return ly

    def interact_features(self, x, ly):
        # concatenate dense and sparse features 
        (batch_size, d) = x.shape 
        print(x.shape)
        #print(ly[0])
        print(ly[0].shape)
        #print(torch.cat([x] + ly, dim=1))
        T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
        print(T.shape)
        # dot product with BMM
        Z = torch.bmm(T, torch.transpose(T, 1, 2))
        print(Z.shape)
        # append dense feature with interactions into a row vector
        _, ni, nj = Z.shape 

        li = torch.tensor([i for i in range(ni) for j in range(i)])
        lj = torch.tensor([j for i in range(nj) for j in range(i)])

        Zflat = Z[:, li, lj]
        # concat features into a row vector
        R = torch.cat([x] + [Zflat], dim=1)
        print(R.shape)

        return R

    #def forward(self, dense_x, lS_o, lS_i):
    #    """
    #    Seq forward single GPU
    #    """

    #    x = self.apply_mlp(dense_x, self.bot_l)

    #    ly = self.apply_emb(lS_o, lS_i, self.emb_l)

    #    z = self.interact_features(x, ly)

    #    p = self.apply_mlp(z, self.top_l)

    #    z = p

    #    return z 


    def forward(self, dense_x, lS_o, lS_i):
        ### prepare model (overwrite) ###
        # expects more ranks than 1 to work.
        # use parallel forward 
        # WARNING: # of devices must be >= batch size in parallel_forward call
        batch_size = dense_x.size()[0]
        print(f"batch_size: {batch_size}")
        ndevices = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if ndevices <= 0:
            sys.exit("This model needs altleast 2 GPUs")

        device_ids = range(ndevices)
        # replicate mlp - data parallelism
        self.bot_l_replicas = replicate(self.bot_l, device_ids)
        self.top_l_replicas = replicate(self.top_l, device_ids)
        # embedding only on gpu 0
        # TODO(rakshithvasudev): check if the placement has no issues
        # large scale embeddings will not work.
        t_list = []
        for k, emb in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            t_list.append(emb.to(d))

        self.emb_l = nn.ModuleList(t_list)

        # scatter dense features (data parallelism)
        # print(dense_x.device)
        dense_x = scatter(dense_x, device_ids, dim=0)

        if (len(self.emb_l) != len(lS_o)) or (len(self.emb_l) != len(lS_i)):
            sys.exit("ERROR: corrupted model input detected in parallel_forward call")
        
        t_list = []
        i_list = [] 

        # distribute offsets and indices to gpu 0
        for k, _ in enumerate(self.emb_l):
            #d = torch.device("cuda:" + str(0))
            d = torch.device("cuda:" + str(k % ndevices))
            t_list.append(lS_o[k].to(d))
            i_list.append(lS_i[k].to(d))
        lS_o = t_list
        lS_i = i_list


        ### compute results in parallel ###
        # bottom mlp
        # WARNING: Note that the self.bot_l is a list of bottom mlp modules
        # that have been replicated across devices, while dense_x is a tuple of dense
        # inputs that has been scattered across devices on the first (batch) dimension.
        # The output is a list of tensors scattered across devices according to the
        # distribution of dense_x.
        x = parallel_apply(self.bot_l_replicas, dense_x, None, device_ids)
        # debug prints
        # print(x)


        # embeddings
        ly = self.apply_emb(lS_o, lS_i, self.emb_l)
        # debug prints
        # print(ly)

        # butterfly shuffle (implemented inefficiently for now)
        # WARNING: Note that at this point we have the result of the embedding lookup
        # for the entire batch on device 0. We would like to obtain partial results
        # corresponding to all embedding lookups, but part of the batch on each device.
        # Therefore, matching the distribution of output of bottom mlp, so that both
        # could be used for subsequent interactions on each device.
        # TODO(rakshithvasudev): Check if this is applicable 
        if len(self.emb_l) != len(ly):
            sys.exit("ERROR: corrupted intermediate result in parallel_forward call")


        t_list = [] 
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:"+ str(k % ndevices))
            # scatter to device 0
            #y = scatter(ly[k], [0], dim=0)
            y = scatter(ly[k], device_ids, dim=0)
            t_list.append(y)



        # adjust list to be ordered per device 
        ly = list(map(lambda y:list(y), zip(*t_list)))
        #print(f"ly: {ly}")

        #debug prints
        #print(ly)

        z = []

        for k in range(ndevices):
            zk = self.interact_features(x[k], ly[k])
            z.append(zk)


        #debug prints
        #print(z)

        # top mlp
        # WARNING: Note that the self.top_l is a list of top mlp modules that
        # have been replicated across devices, while z is a list of interaction results
        # that by construction are scattered across devices on the first (batch) dim.
        # The output is a list of tensors scattered across devices according to the
        # distribution of z.
        p = parallel_apply(self.top_l_replicas, z, None, device_ids)


        ### gather the distributed results ###
        p0 = gather(p, 0, dim=0)

        
        z0 = p0

        return z0


    def dlrm_wrap(X, lS_o, lS_i, device):
        return dlrm













if __name__=="__main__":

    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )

    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument(
        "--arch-embedding-size", type=dash_separated_ints, default="4-3-2"
    )
    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=dash_separated_ints, default="4-2-1")
    #parser.add_argument("--arch-mlp-bot", type=dash_separated_ints, default="13-512-256-128")
    #parser.add_argument("--arch-mlp-top", type=dash_separated_ints, default="1024-1024-512-256-1")
    parser.add_argument(
        "--arch-interaction-op", type=str, choices=["dot", "cat"], default="dot"
    )
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    parser.add_argument("--weighted-pooling", type=str, default=None)
    # embedding table options
    parser.add_argument("--md-flag", action="store_true", default=False)
    parser.add_argument("--md-threshold", type=int, default=200)
    parser.add_argument("--md-temperature", type=float, default=0.3)
    parser.add_argument("--md-round-dims", action="store_true", default=False)
    parser.add_argument("--qr-flag", action="store_true", default=False)
    parser.add_argument("--qr-threshold", type=int, default=200)
    parser.add_argument("--qr-operation", type=str, default="mult")
    parser.add_argument("--qr-collisions", type=int, default=4)
    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str, default="mse")  # or bce or wbce
    parser.add_argument(
        "--loss-weights", type=dash_separated_floats, default="1.0-1.0"
    )  # for wbce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--data-size", type=int, default=50000000)
    parser.add_argument("--num-batches", type=int, default=1000)
    parser.add_argument(
        "--data-generation", type=str, default="random"
    )  # synthetic or dataset
    parser.add_argument(
        "--rand-data-dist", type=str, default="uniform"
    )  # uniform or gaussian
    parser.add_argument("--rand-data-min", type=float, default=0)
    parser.add_argument("--rand-data-max", type=float, default=1)
    parser.add_argument("--rand-data-mu", type=float, default=-1)
    parser.add_argument("--rand-data-sigma", type=float, default=1)
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--memory-map", action="store_true", default=False)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=64*64)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument(
        "--dataset-multiprocessing",
        action="store_true",
        default=False,
        help="The Kaggle dataset can be multiprocessed in an environment \
                        with more than 7 CPU cores and more than 20 GB of memory. \n \
                        The Terabyte dataset can be multiprocessed in an environment \
                        with more than 24 CPU cores and at least 1 TB of memory.",
    )
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # quantize
    parser.add_argument("--quantize-mlp-with-bit", type=int, default=32)
    parser.add_argument("--quantize-emb-with-bit", type=int, default=32)
    # onnx
    parser.add_argument("--save-onnx", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # distributed
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dist-backend", type=str, default="")
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-workers", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--print-wall-time", action="store_true", default=False)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--plot-compute-graph", action="store_true", default=False)
    parser.add_argument("--tensor-board-filename", type=str, default="run_kaggle_pt")
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    # mlperf logging (disables other output and stops early)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)
    # stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
    parser.add_argument("--mlperf-acc-threshold", type=float, default=0.0)
    # stop at target AUC Terabyte (no subsampling) 0.8025
    parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader", action="store_true", default=False)
    parser.add_argument("--mlperf-bin-shuffle", action="store_true", default=False)
    # mlperf gradient accumulation iterations
    parser.add_argument("--mlperf-grad-accum-iter", type=int, default=1)
    # LR policy
    parser.add_argument("--lr-num-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-start-step", type=int, default=0)
    parser.add_argument("--lr-num-decay-steps", type=int, default=0)

    global args
    global nbatches
    global nbatches_test
    global writer
    args = parser.parse_args()



    m_spa=26
    # ln_emb
    vocab_size = np.array([128])
    #vocab_size = np.fromstring("1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000", sep="-", dtype=int)
    print(vocab_size)
    bottom_mlp = np.array([13,512, 256, 128, m_spa])
    #TODO(rakshithvasudev): update the adjusted input to dynamic
    adjusted_input = 27
    top_mlp = np.array([adjusted_input, 1024, 1024, 512, 256, 1])



    #embedding = dlrm.create_emb(m_spa, vocab_size)



    m_den = 13
    train_data, train_loader, test_data, test_loader = make_random_data_and_loader(
            args, vocab_size, m_den,
            offset_to_length_converter=False,
            )

    #print(type(train_loader))
    #print(len(train_loader))
    it = iter(train_loader)
    #batch =next(it) 
    
    


    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    torch.cuda.manual_seed_all(args.numpy_rand_seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda", 0)
    ngpus = torch.cuda.device_count()  # 1
    print("Using {} GPU(s)...".format(ngpus))
    
    def time_wrap():
        torch.cuda.synchronize()
        return time.time()

    
    torch.cuda.synchronize()

    dlrm = DlrmSmall(m_spa, vocab_size, bottom_mlp, top_mlp)

    #dlrm = nn.DataParallel(dlrm)
    
    dlrm = dlrm.to(device)
    
    def dlrm_wrap(X, lS_o, lS_i, device):
        return dlrm(X.to(device), 
                [S_o.to(device) for S_o in lS_o],
                [S_i.to(device) for S_i in lS_i],
                )
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.SGD(dlrm.parameters(), lr=args.learning_rate)

    def loss_fn_wrap(Z, T, device):
        return loss_fn(Z, T.to(device))

    #dlrm.apply_emb(batch[1], batch[2], embedding)
    #dlrm.forward(batch[0], batch[1], batch[2])

    #for j in range(0, 64):
    #    batch = next(it)
    #    t1 = time_wrap()
    #    Z = dlrm_wrap(batch[0], batch[1], batch[2], device)
    #    print(f"forward: {Z}")
    #    print(f"forward: {Z.shape}")



    for j in range(0, 10000):
        batch = next(it)
        t1 = time_wrap()
        Z = dlrm_wrap(batch[0], batch[1], batch[2], device)
        print(f"forward: {Z}")
        print(f"forward: {Z.shape}")

        E = loss_fn_wrap(Z, batch[3], device)

        # compute loss and accuracy
        L = E.detach().cpu().numpy()  # numpy array
        S = Z.detach().cpu().numpy()  # numpy array
        T = batch[3].detach().cpu().numpy()  # numpy array
        mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
        A = np.sum((np.round(S, 0) == T).astype(np.uint8)) / mbs

        optimizer.zero_grad()
        optimizer.step()
        t2 = time_wrap()

        print(f"time to train: {t2 -t1}s")
