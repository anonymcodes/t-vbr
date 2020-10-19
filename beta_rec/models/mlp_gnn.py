import torch
import os
from beta_rec.models.gcn import GCN_S
from beta_rec.models.torch_engine import Engine
from beta_rec.utils.common_util import print_dict_as_table


class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.num_users = config["n_users"]
        self.num_items = config["n_items"]
        self.emb_dim = config["emb_dim"]
        self.layers = [self.emb_dim * 2] + config["layers"]
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.emb_dim
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.emb_dim
        )

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(
            in_features=self.layers[-1], out_features=1
        )
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat(
            [user_embedding, item_embedding], dim=-1
        )  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
            # vector = torch.nn.BatchNorm1d()(vector)
            # vector = torch.nn.Dropout(p=0.5)(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def predict(self, user_indices, item_indices):
        user_indices = torch.LongTensor(user_indices).to(self.device)
        item_indices = torch.LongTensor(item_indices).to(self.device)
        with torch.no_grad():
            return self.forward(user_indices, item_indices)

    def init_weight(self):
        pass


class MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""

    def __init__(self, config, gcn_config=None):
        self.model = MLP(config)
        print_dict_as_table(config, tag="MLP config")
        self.gcn_config = gcn_config
        super(MLPEngine, self).__init__(config)
        self.model.to(self.device)
        if gcn_config is not None:
            self.load_pretrain_weights()

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, "model"), "Please specify the exact model !"
        users, items, ratings = (
            users.to(self.device),
            items.to(self.device),
            ratings.to(self.device),
        )
        self.optimizer.zero_grad()
        ratings_pred = self.model.forward(users, items)
        loss = self.loss(ratings_pred.view(-1), ratings)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            total_loss += loss
        print("[Training Epoch {}], Loss {}".format(epoch_id, total_loss))
        self.writer.add_scalar("model/loss", total_loss, epoch_id)

    def load_pretrain_weights(self):
        """Loading weights from trained GMF model"""
        gcn_model = GCN_S(self.gcn_config)
        self.resume_checkpoint(
            os.path.join(self.config["model_save_dir"], self.config["pretrain_gcn"]),
            gcn_model,
        )
        self.model.embedding_user.weight.data = gcn_model.user_embedding.weight.data
        self.model.embedding_item.weight.data = gcn_model.item_embedding.weight.data
