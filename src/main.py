"""Running CapsGNN."""

from utils import tab_printer
from capsgnn import CapsGNNTrainer
from param_parser import parameter_parser

def main():
    """
    Parsing command line parameters, processing graphs, fitting a CapsGNN.
    """
    args = parameter_parser()
    tab_printer(args)
    model = CapsGNNTrainer(args)
    model.fit()
    model.score()
    model.save_predictions()

if __name__ == "__main__":
    main()
