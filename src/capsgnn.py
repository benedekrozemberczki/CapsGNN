"""CapsGNN Trainer."""

import glob
import json
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from torch_geometric.nn import GCNConv
from utils import create_numeric_mapping
from layers import ListModule, PrimaryCapsuleLayer, Attention, SecondaryCapsuleLayer
from layers import margin_loss

class CapsGNN(torch.nn.Module):
    """
    An implementation of themodel described in the following paper:
    https://openreview.net/forum?id=Byl8BnRcYm
    """
    def __init__(self, args, number_of_features, number_of_targets):
        super(CapsGNN, self).__init__()
        """
        :param args: Arguments object.
        :param number_of_features: Number of vertex features.
        :param number_of_targets: Number of classes.
        """
        self.args = args
        self.number_of_features = number_of_features
        self.number_of_targets = number_of_targets
        self._setup_layers()

    def _setup_base_layers(self):
        """
        Creating GCN layers.
        """
        self.base_layers = [GCNConv(self.number_of_features, self.args.gcn_filters)]
        for _ in range(self.args.gcn_layers-1):
            self.base_layers.append(GCNConv(self.args.gcn_filters, self.args.gcn_filters))
        self.base_layers = ListModule(*self.base_layers)

    def _setup_primary_capsules(self):
        """
        Creating primary capsules.
        """
        self.first_capsule = PrimaryCapsuleLayer(in_units=self.args.gcn_filters,
                                                 in_channels=self.args.gcn_layers,
                                                 num_units=self.args.gcn_layers,
                                                 capsule_dimensions=self.args.capsule_dimensions)

    def _setup_attention(self):
        """
        Creating attention layer.
        """
        self.attention = Attention(self.args.gcn_layers*self.args.capsule_dimensions,
                                   self.args.inner_attention_dimension)

    def _setup_graph_capsules(self):
        """
        Creating graph capsules.
        """
        self.graph_capsule = SecondaryCapsuleLayer(self.args.gcn_layers,
                                                   self.args.capsule_dimensions,
                                                   self.args.number_of_capsules,
                                                   self.args.capsule_dimensions)

    def _setup_class_capsule(self):
        """
        Creating class capsules.
        """
        self.class_capsule = SecondaryCapsuleLayer(self.args.capsule_dimensions,
                                                   self.args.number_of_capsules,
                                                   self.number_of_targets,
                                                   self.args.capsule_dimensions)

    def _setup_reconstruction_layers(self):
        """
        Creating histogram reconstruction layers.
        """
        self.reconstruction_layer_1 = torch.nn.Linear(self.number_of_targets*self.args.capsule_dimensions,
                                                      int((self.number_of_features*2)/3))

        self.reconstruction_layer_2 = torch.nn.Linear(int((self.number_of_features*2)/3),
                                                      int((self.number_of_features*3)/2))

        self.reconstruction_layer_3 = torch.nn.Linear(int((self.number_of_features*3)/2),
                                                      self.number_of_features)

    def _setup_layers(self):
        """
        Creating layers of model.
        1. GCN layers.
        2. Primary capsules.
        3. Attention
        4. Graph capsules.
        5. Class capsules.
        6. Reconstruction layers.
        """
        self._setup_base_layers()
        self._setup_primary_capsules()
        self._setup_attention()
        self._setup_graph_capsules()
        self._setup_class_capsule()
        self._setup_reconstruction_layers()

    def calculate_reconstruction_loss(self, capsule_input, features):
        """
        Calculating the reconstruction loss of the model.
        :param capsule_input: Output of class capsule.
        :param features: Feature matrix.
        :return reconstrcution_loss: Loss of reconstruction.
        """

        v_mag = torch.sqrt((capsule_input**2).sum(dim=1))
        _, v_max_index = v_mag.max(dim=0)
        v_max_index = v_max_index.data

        capsule_masked = torch.autograd.Variable(torch.zeros(capsule_input.size()))
        capsule_masked[v_max_index, :] = capsule_input[v_max_index, :]
        capsule_masked = capsule_masked.view(1, -1)

        feature_counts = features.sum(dim=0)
        feature_counts = feature_counts/feature_counts.sum()

        reconstruction_output = torch.nn.functional.relu(self.reconstruction_layer_1(capsule_masked))
        reconstruction_output = torch.nn.functional.relu(self.reconstruction_layer_2(reconstruction_output))
        reconstruction_output = torch.softmax(self.reconstruction_layer_3(reconstruction_output), dim=1)
        reconstruction_output = reconstruction_output.view(1, self.number_of_features)
        reconstruction_loss = torch.sum((features-reconstruction_output)**2)
        return reconstruction_loss

    def forward(self, data):
        """
        Forward propagation pass.
        :param data: Dictionary of tensors with features and edges.
        :return class_capsule_output: Class capsule outputs.
        """
        features = data["features"]
        edges = data["edges"]
        hidden_representations = []

        for layer in self.base_layers:
            features = torch.nn.functional.relu(layer(features, edges))
            hidden_representations.append(features)

        hidden_representations = torch.cat(tuple(hidden_representations))
        hidden_representations = hidden_representations.view(1, self.args.gcn_layers, self.args.gcn_filters, -1)
        first_capsule_output = self.first_capsule(hidden_representations)
        first_capsule_output = first_capsule_output.view(-1, self.args.gcn_layers*self.args.capsule_dimensions)
        rescaled_capsule_output = self.attention(first_capsule_output)
        rescaled_first_capsule_output = rescaled_capsule_output.view(-1, self.args.gcn_layers,
                                                                     self.args.capsule_dimensions)
        graph_capsule_output = self.graph_capsule(rescaled_first_capsule_output)
        reshaped_graph_capsule_output = graph_capsule_output.view(-1, self.args.capsule_dimensions,
                                                                  self.args.number_of_capsules)
        class_capsule_output = self.class_capsule(reshaped_graph_capsule_output)
        class_capsule_output = class_capsule_output.view(-1, self.number_of_targets*self.args.capsule_dimensions)
        class_capsule_output = torch.mean(class_capsule_output, dim=0).view(1,
                                                                            self.number_of_targets,
                                                                            self.args.capsule_dimensions)
        recon = class_capsule_output.view(self.number_of_targets, self.args.capsule_dimensions)
        reconstruction_loss = self.calculate_reconstruction_loss(recon, data["features"])
        return class_capsule_output, reconstruction_loss


class CapsGNNTrainer(object):
    """
    CapsGNN training and scoring.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.setup_model()

    def enumerate_unique_labels_and_targets(self):
        """
        Enumerating the features and targets in order to setup weights later.
        """
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
        """
        Enumerating labels and initializing a CapsGNN.
        """
        self.enumerate_unique_labels_and_targets()
        self.model = CapsGNN(self.args, self.number_of_features, self.number_of_targets)

    def create_batches(self):
        """
        Batching the graphs for training.
        """
        self.batches = []
        for i in range(0, len(self.train_graph_paths), self.args.batch_size):
            self.batches.append(self.train_graph_paths[i:i+self.args.batch_size])

    def create_data_dictionary(self, target, edges, features):
        """
        Creating a data dictionary.
        :param target: Target vector.
        :param edges: Edge list tensor.
        :param features: Feature tensor.
        """
        to_pass_forward = dict()
        to_pass_forward["target"] = target
        to_pass_forward["edges"] = edges
        to_pass_forward["features"] = features
        return to_pass_forward

    def create_target(self, data):
        """
        Target createn based on data dicionary.
        :param data: Data dictionary.
        :return : Target vector.
        """
        return  torch.FloatTensor([0.0 if i != data["target"] else 1.0 for i in range(self.number_of_targets)])

    def create_edges(self, data):
        """
        Create an edge matrix.
        :param data: Data dictionary.
        :return : Edge matrix.
        """
        edges = [[edge[0], edge[1]] for edge in data["edges"]]
        edges = edges + [[edge[1], edge[0]] for edge in data["edges"]]
        return torch.t(torch.LongTensor(edges))

    def create_features(self, data):
        """
        Create feature matrix.
        :param data: Data dictionary.
        :return features: Matrix of features.
        """
        features = np.zeros((len(data["labels"]), self.number_of_features))
        node_indices = [node for node in range(len(data["labels"]))]
        feature_indices = [self.feature_map[label] for label in data["labels"].values()]
        features[node_indices, feature_indices] = 1.0
        features = torch.FloatTensor(features)
        return features

    def create_input_data(self, path):
        """
        Creating tensors and a data dictionary with Torch tensors.
        :param path: path to the data JSON.
        :return to_pass_forward: Data dictionary.
        """
        data = json.load(open(path))
        target = self.create_target(data)
        edges = self.create_edges(data)
        features = self.create_features(data)
        to_pass_forward = self.create_data_dictionary(target, edges, features)
        return to_pass_forward

    def fit(self):
        """
        Training a model on the training set.
        """
        print("\nTraining started.\n")
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)

        for _ in tqdm(range(self.args.epochs), desc="Epochs: ", leave=True):
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
                    prediction, reconstruction_loss = self.model(data)
                    loss = margin_loss(prediction,
                                       data["target"],
                                       self.args.lambd)
                    loss = loss+self.args.theta*reconstruction_loss
                    accumulated_losses = accumulated_losses + loss
                accumulated_losses = accumulated_losses/len(batch)
                accumulated_losses.backward()
                optimizer.step()
                losses = losses + accumulated_losses.item()
                average_loss = losses/(step + 1)
                self.steps.set_description("CapsGNN (Loss=%g)" % round(average_loss, 4))

    def score(self):
        """
        Scoring on the test set.
        """
        print("\n\nScoring.\n")
        self.model.eval()
        self.predictions = []
        self.hits = []
        for path in tqdm(self.test_graph_paths):
            data = self.create_input_data(path)
            prediction, _ = self.model(data)
            prediction_mag = torch.sqrt((prediction**2).sum(dim=2))
            _, prediction_max_index = prediction_mag.max(dim=1)
            prediction = prediction_max_index.data.view(-1).item()
            self.predictions.append(prediction)
            self.hits.append(data["target"][prediction] == 1.0)

        print("\nAccuracy: " + str(round(np.mean(self.hits), 4)))

    def save_predictions(self):
        """
        Saving the test set predictions.
        """
        identifiers = [path.split("/")[-1].strip(".json") for path in self.test_graph_paths]
        out = pd.DataFrame()
        out["id"] = identifiers
        out["predictions"] = self.predictions
        out.to_csv(self.args.prediction_path, index=None)
