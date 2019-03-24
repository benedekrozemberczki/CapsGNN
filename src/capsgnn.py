import torch
import glob
import json
import random
from tqdm import tqdm, trange
import numpy as np
from utils import create_numeric_mapping
from torch_geometric.nn import GCNConv
from layers import ListModule
import pandas as pd

class CapsGNN(torch.nn.Module):

    def __init__(self, args, number_of_features, number_of_targets):
        super(CapsGNN, self).__init__()
        self.args = args
        self.number_of_features = number_of_features
        self.number_of_targets = number_of_targets
        self._setup_layers()

    def _setup_layers(self):
        self.base_layers = [GCNConv(self.number_of_features, self.args.gcn_filters)]
        for layer in range(self.args.gcn_layers-1):
            self.base_layers.append(GCNConv( self.args.gcn_filters, self.args.gcn_filters))
        self.base_layers = ListModule(*self.base_layers)
        self.out = torch.nn.Linear(self.args.gcn_filters, self.number_of_targets)

    def forward(self, data):
        features = data["features"]
        edges = data["edges"]
        hidden_representations = []
        for layer in self.base_layers:
            features = layer(features, edges)
            hidden_representations.append(features)
        features = torch.mean(features, dim = 0)
        prediction = self.out(features)
        prediction = torch.nn.functional.log_softmax(prediction, dim=0).view(1,-1)

        return prediction

class CapsGNNTrainer(object):

    def __init__(self,args):
        self.args = args
        self.setup_model()

    def enumerate_unique_labels_and_targets(self):
        print("\nEnumerating feature and target values.\n")
        ending = "*.json"

        self.train_graph_paths = glob.glob(self.args.train_graph_folder+ending)
        self.test_graph_paths = glob.glob(self.args.test_graph_folder+ending)
    
        graph_paths = self.train_graph_paths + self.test_graph_paths

        targets = set()
        features = set()
        for path in tqdm(graph_paths):
            data = json.load(open(path))
            targets = targets.union(set([data["target"]]))
            features = features.union(set(data["labels"]))

        self.target_map = create_numeric_mapping(targets)
        self.feature_map = create_numeric_mapping(features)

        self.number_of_features = len(self.feature_map)
        self.number_of_targets = len(self.target_map)
    
    def setup_model(self):
        self.enumerate_unique_labels_and_targets()
        self.model = CapsGNN(self.args, self.number_of_features, self.number_of_targets)

    def create_batches(self):
        self.batches = [self.train_graph_paths[i:i + self.args.batch_size] for i in range(0,len(self.train_graph_paths), self.args.batch_size)]

    def create_data_dictionary(self,target, edges, features):
        to_pass_forward = dict()
        to_pass_forward["target"] = target
        to_pass_forward["edges"] = edges
        to_pass_forward["features"] = features
        return to_pass_forward


    def create_target(self,data):
        return torch.LongTensor([data["target"]])

    def create_edges(self,data):
        return torch.t(torch.LongTensor(data["edges"]))

    def create_features(self,data):
        features = np.zeros((len(data["labels"]), self.number_of_features))
        node_indices = [node for node in range(len(data["labels"]))]
        feature_indices = [self.feature_map[label] for label in data["labels"].values()] 
        features[node_indices,feature_indices] = 1.0
        features = torch.FloatTensor(features)
        return features

    def create_input_data(self, path):
        data = json.load(open(path))
        target = self.create_target(data)
        edges = self.create_edges(data)
        features = self.create_features(data)
        to_pass_forward = self.create_data_dictionary(target, edges, features)
        return to_pass_forward

    def fit(self):
        print("\nTraining started.\n")
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        for epoch in tqdm(range(self.args.epochs), desc = "Epochs: ", leave = True):
            random.shuffle(self.train_graph_paths)
            self.create_batches()
            losses = 0       
            self.steps = trange(len(self.batches), desc="Loss")
            for step in self.steps:
                accumulated_losses = 0
                optimizer.zero_grad()
                batch = self.batches[step]
                for path in batch:
                    data = self.create_input_data(path)
                    prediction = self.model(data)
                    loss = torch.nn.functional.nll_loss(prediction, data["target"])
                    accumulated_losses = accumulated_losses + loss
                accumulated_losses = accumulated_losses/len(path)
                accumulated_losses.backward()
                optimizer.step()
                losses = losses+accumulated_losses.item()
                average_loss = losses/(step+1)
                self.steps.set_description("CapsGNN (Loss=%g)" % round(average_loss,4))

    def score(self):
        print("\n\nScoring.\n")
        self.model.eval()
        self.predictions = []
        self.hits = []
        for path in tqdm(self.test_graph_paths):
            data = self.create_input_data(path)
            prediction = self.model(data)
            prediction = torch.argmax(prediction).item()
            self.predictions.append(prediction)
            self.hits.append(prediction==data["target"])
        print("\nAccuracy: " + str(round(np.mean(self.hits),4)))

    def save_predictions(self):
        identifiers = [path.split("/")[-1].strip(".json") for path in self.test_graph_paths]
        out = pd.DataFrame()
        out["id"] = identifiers
        out["predictions"] = self.predictions
        out.to_csv(self.args.prediction_path, index = None)

