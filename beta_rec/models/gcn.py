import torch
from beta_rec.utils.common_util import print_dict_as_table
from beta_rec.models.torch_engine import Engine
import torch.nn.functional as F
import torch.nn as nn


class GCN_S(torch.nn.Module):
    """Initialize embedding with the single graph.

    """

    def __init__(self, config):
        super(GCN_S, self).__init__()
        self.config = config
        self.n_users = config["n_users"]
        self.n_items = config["n_items"]
        self.emb_dim = config["emb_dim"]
        self.layers = config["gcn_config"]["layers"]
        self.n_layers = len(self.layers)
        self.dropout = nn.ModuleList()
        self.u_gcn_weights = nn.ModuleList()
        self.i_gcn_weights = nn.ModuleList()
        self.layers = [self.emb_dim] + self.layers
        self.dropout_list = list(config["gcn_config"]["mess_dropout"])
        # Create GNN layers

        for i in range(self.n_layers):
            self.u_gcn_weights.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            self.i_gcn_weights.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            self.dropout.append(nn.Dropout(self.dropout_list[i]))

        self.embedding_user = nn.Embedding(self.n_users, self.emb_dim)
        self.embedding_item = nn.Embedding(self.n_items, self.emb_dim)
        self.init_emb()

    def init_emb(self):
        # Initialise users and items' embeddings
        nn.init.normal_(self.embedding_user.weight, std=0.01)
        nn.init.normal_(self.embedding_user.weight, std=0.01)

    def forward(self, user_fea_norm_adj, item_fea_norm_adj):
        """ Perform GNN function on users and item embeddings
        Args:
            user_fea_norm_adj (torch sparse tensor): the norm adjacent matrix of the user-user similarity matrix
            item_fea_norm_adj (torch sparse tensor): the norm adjacent matrix of the item-item similarity matrix
        Returns:
            u_embeddings (tensor): processed user embeddings
            i_embeddings (tensor): processed item embeddings
        """
        u_embeddings = self.embedding_user.weight
        i_embeddings = self.embedding_item.weight

        for i in range(self.n_layers):
            u_embeddings = torch.sparse.mm(user_fea_norm_adj, u_embeddings)
            u_embeddings = F.leaky_relu(self.u_gcn_weights[i](u_embeddings))
            u_embeddings = self.dropout[i](u_embeddings)
            u_embeddings = F.normalize(u_embeddings, p=2, dim=1)

            i_embeddings = torch.sparse.mm(item_fea_norm_adj, i_embeddings)
            i_embeddings = F.leaky_relu(self.i_gcn_weights[i](i_embeddings))
            i_embeddings = self.dropout[i](i_embeddings)
            i_embeddings = F.normalize(i_embeddings, p=2, dim=1)

        return u_embeddings, i_embeddings

    def predict(self, users, items):
        """ Model prediction: dot product of users and items embeddings
        Args:
            users (int):  user id
            items (int):  item id
        Return:
            scores (int): dot product
        """
        users_t = torch.tensor(users, dtype=torch.int64, device=self.device)
        items_t = torch.tensor(items, dtype=torch.int64, device=self.device)

        with torch.no_grad():
            scores = torch.mul(
                self.embedding_user(users_t), self.embedding_item(items_t)
            ).sum(dim=1)
        return scores


class GCN_SEngine(Engine):
    # A class includes train an epoch and train a batch of NGCF

    def __init__(self, config):
        self.config = config
        print_dict_as_table(config, tag="GCN config")
        self.model = GCN_S(config)
        self.regs = config["gcn_config"]["regs"]  # reg is the regularisation
        self.decay = self.regs[0]
        self.batch_size = config["batch_size"]
        self.num_batch = config["num_batch"]
        self.user_fea_norm_adj = config["user_fea_norm_adj"]
        self.item_fea_norm_adj = config["item_fea_norm_adj"]
        super(GCN_SEngine, self).__init__(config)

    def train_single_batch(self, batch_data):
        """
        Args:
            batch_data (list): batch users, positive items and negative items
        Return:
            loss (float): batch loss
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.optimizer.zero_grad()
        ua_embeddings, ia_embeddings = self.model.forward(
            self.user_fea_norm_adj, self.item_fea_norm_adj
        )

        batch_users, pos_items, neg_items = batch_data

        u_g_embeddings = ua_embeddings[batch_users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
        )

        batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss

        batch_loss.backward()
        self.optimizer.step()
        loss = batch_loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        """ Generate batch data for each batch
        Args:
            epoch_id (int):
            user (list)
            pos_i (list):
            neg_i (list):
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0.0
        n_batch = self.num_batch
        for idx in range(n_batch):
            batch_data = train_loader.sample(self.batch_size)
            loss = self.train_single_batch(batch_data)
            total_loss += loss
        print("[Training Epoch {}], Loss {}".format(epoch_id, loss))
        self.writer.add_scalar("model/loss", total_loss, epoch_id)

    def bpr_loss(self, users, pos_items, neg_items):
        # Calculate BPR loss
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = (
            1.0 / 2 * (users ** 2).sum()
            + 1.0 / 2 * (pos_items ** 2).sum()
            + 1.0 / 2 * (neg_items ** 2).sum()
        )
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss
