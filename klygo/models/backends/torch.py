from typing import Any
from ..registry import registry
from ..exceptions import ModelLoadError, BackendExecutionError

@registry.register_backend("torch")
class TorchBackend:
    def __init__(self, raw_model: Any, device: str = "cpu", **kwargs):
        import torch
        self.model = raw_model
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def load(cls, model_path: str, device: str = "cpu", **kwargs):
        """Loads PyTorch model. If model_class is provided, loads state dict; otherwise, loads as serialized TorchScript model."""
        import torch
        try:
            model_class = kwargs.get("model_class")
            
            if model_class is not None:
                # Load state dict
                loaded = torch.load(model_path, map_location="cpu")
                state_dict = loaded.get("state_dict", loaded)
                raw_model = model_class()
                raw_model.load_state_dict(state_dict)
            else:
                # Default: load as serialized TorchScript model (does not require python code class definitions)
                raw_model = torch.jit.load(model_path, map_location="cpu")
                
            return cls(raw_model, device=device, **kwargs)
        except Exception as e:
            raise ModelLoadError(f"Failed to load PyTorch model from {model_path}: {e}")

    @property
    def native(self):
        return self.model

    def predict(self, images: Any, **kwargs) -> Any:
        """Executes forward pass. Expects raw tensors or PIL Images converted to tensors."""
        import torch
        self.model.eval()
        
        # If input is already a PyTorch Tensor, directly run inference
        if isinstance(images, torch.Tensor):
            with torch.no_grad():
                return self.model(images.to(self.device))
                
        # If input is a list of PyTorch Tensors, stack them directly
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], torch.Tensor):
            batch_tensor = torch.stack(images).to(self.device)
            with torch.no_grad():
                return self.model(batch_tensor)
                
        # Convert PIL images to tensors if needed
        from klygo import media
        tensors = []
        for img in images:
            tensors.append(media.to_tensor(img).to(self.device))
        
        # Batch tensors
        batch_tensor = torch.stack(tensors)
        with torch.no_grad():
            outputs = self.model(batch_tensor)
        return outputs

    def train(self, train_loader, optimizer, loss_fn, epochs=10, **kwargs) -> Any:
        """Runs a standard PyTorch training loop on the wrapped module in-place."""
        import torch
        self.model.to(self.device)
        self.model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
        return self.model

    def evaluate(self, val_loader, loss_fn, **kwargs) -> dict:
        """Evaluates model performance on validation data loader."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
        return {
            "loss": total_loss / len(val_loader),
            "accuracy": correct / total
        }

    def save(self, path: str):
        """Saves the state_dict of the model to disk."""
        try:
            torch.save(self.model.state_dict(), path)
        except Exception as e:
            raise BackendExecutionError(f"Failed to save state_dict to {path}: {e}")
