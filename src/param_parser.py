import argparse

def parameter_parser():
    """
    A method to parse up command line parameters. By default it learns on the Watts-Strogatz dataset.
    The default hyperparameters give good results without cross-validation.
    """
    parser = argparse.ArgumentParser(description="Run CapsGNN.")
	
    parser.add_argument("--train-graph-folder",
                        nargs="?",
                        default="./input/train/",
	                help="Training graphs folder.")

    parser.add_argument("--test-graph-folder",
                        nargs="?",
                        default="./input/test/",
	                help="Testing graphs folder.")

    parser.add_argument("--prediction-path",
                        nargs="?",
                        default="./output/watts_predictions.csv",
	                help="Path to store the predicted graph labels.")

    parser.add_argument("--epochs",
                        type=int,
                        default=100,
	                help="Number of training epochs. Default is 100.")

    parser.add_argument("--batch-size",
                        type=int,
                        default=32,
	                help="Number of graphs processed per batch. Default is 32.")

    parser.add_argument("--gcn-filters",
                        type=int,
                        default=20,
	                help="Number of Graph Convolutional filters. Default is 20.")

    parser.add_argument("--gcn-layers",
                        type=int,
                        default=2,
	                help="Number of Graph Convolutional Layers. Default is 2.")

    parser.add_argument("--inner-attention-dimension",
                        type=int,
                        default=20,
	                help="Number of Attention Neurons. Default is 20.")

    parser.add_argument("--capsule-dimensions",
                        type=int,
                        default=8,
	                help="Capsule dimensions. Default is 8.")

    parser.add_argument("--number-of-capsules",
                        type=int,
                        default=8,
	                help="Number of capsules per layer. Default is 8.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=10**-6,
	                help="Weight decay. Default is 10^-6.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.01,
	                help="Learning rate. Default is 0.01.")

    parser.add_argument("--lambd",
                        type=float,
                        default=0.5,
	                help="Loss combination weight. Default is 0.5.")

    parser.add_argument("--theta",
                        type=float,
                        default=0.1,
	                help="Reconstruction loss weight. Default is 0.1.")

    return parser.parse_args()
