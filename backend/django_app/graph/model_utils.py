import torch
from torch_geometric.utils import from_networkx
import networkx as nx
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast

from .gnn_model import DeepReasoningGNN  # Import the GNN model

# Load Transformer model (if needed)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
transformer_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

projection_layer = torch.nn.Linear(768, 64)
projection_layer.eval()  # Set to evaluation mode

# Function to get embeddings from the transformer
def get_projected_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = transformer_model(**inputs).last_hidden_state.mean(dim=1)
        print(embeddings)
        projected_embedding = projection_layer(embeddings)
    return projected_embedding.numpy()


# Model preparation
def load_model():
    model = DeepReasoningGNN(in_channels=64, hidden_channels=128, out_channels=64, num_layers=4)
    model.load_state_dict(torch.load(r'C:\Users\hamdi\Documents\django_projects\PI\backend\django_app\graph\model\deep_reasoning_gnn_model.pth'))
    model.eval()
    return model

# Inference function
def infer_reasoning(model, query_text, graph_data, top_k=10):
    # Step 1: Get the projected embedding for the query text (using the model's projection layer)
    query_embedding = get_projected_embedding(query_text)  # Get query embedding

    # Step 2: Get the node embeddings from the graph_data (which has embeddings loaded in `x`)
    node_embeddings_matrix = np.array([
    np.array(ast.literal_eval(graph_data.nodes[node]['embedding'])) if isinstance(graph_data.nodes[node]['embedding'], str) 
    else np.array(graph_data.nodes[node]['embedding'])  # In case it's already a list
    for node in graph_data.nodes
])
  # Convert PyTorch tensor to NumPy array for similarity calculation

    # Step 3: Calculate cosine similarity between the query embedding and all node embeddings
    similarities = cosine_similarity(query_embedding, node_embeddings_matrix).flatten()

    # Step 4: Get the top-k most similar nodes based on cosine similarity
    top_indices = similarities.argsort()[-top_k:][::-1]  # Get indices of top-k similarities
    node_names = list(graph_data.nodes)  # List of node names
    
    top_nodes = [(node_names[i], similarities[i]) for i in top_indices]  # Get nodes and their similarity scores

    return top_nodes
