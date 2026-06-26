import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    class GCNConv(nn.Module): pass

class SympatheticFlareGNN(nn.Module):
    """
    Graph Neural Network for analyzing sympathetic flares between Active Regions.
    Nodes: Active Regions (features = SHARP magnetic params)
    Edges: Physical distances / magnetic connectivity on the solar disk.
    
    For the hackathon prototype, if only 1 AR is present, it acts as a self-loop graph.
    """
    def __init__(self, in_channels: int = 21, hidden_channels: int = 64, out_channels: int = 128, dropout: float = 0.1):
        super().__init__()
        self.out_channels = out_channels
        if not HAS_PYG:
            print("WARNING: torch_geometric not installed. GNN disabled.")
            self.enabled = False
            return
            
        self.enabled = True
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.proj = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges]
        batch: [num_nodes] mapping from node to graph in the batch
        
        Returns:
            embedding: [batch_size, out_channels]
        """
        if not self.enabled:
            # Fallback if PyG is missing
            batch_size = int(batch.max().item() + 1) if batch.numel() > 0 else 1
            return torch.zeros((batch_size, self.out_channels), device=x.device)
            
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.gelu(x)
        
        # Global pooling (aggregate all ARs in the sun disk snapshot into one vector)
        x = global_mean_pool(x, batch)
        
        x = self.proj(x)
        return x
