import torch.nn as nn

class GenerativeModel(nn.Module):
    """
    Abstract base class for generative models.
    It holds the neural network and defines the interface
    for training sampling and loss calculation.
    """
    def __init__(self, network: nn.Module, infer: bool = False):
        super().__init__()
        self.network = network # The U-Net or Transformer
        self.infer = infer
        if self.infer:
            self.network.eval()
        else:
            self.network.train()

    def get_training_objective(self, *args, **kwargs):
        """
        Returns the tensors needed to compute the loss.
        e.g., (predicted_vector_field, target_vector_field)
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        if self.infer:
            return self.sample(*args, **kwargs)
        else:
            return self.get_training_objective(*args, **kwargs)

    def sample(self, *args, **kwargs):
        """
        Defines the generation process (e.g., the ODE/SDE).
        This is often paired with a sampler class.
        """
        raise NotImplementedError
