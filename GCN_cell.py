import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load the CSV file
file_path = '../PM25.csv'
pm25_data = pd.read_csv(file_path)

# Normalize the data (each row represents a cell's time series data)
scaler = StandardScaler()
pm25_data_normalized = scaler.fit_transform(pm25_data)

# Create kNN graph (k=5)
k = 5
knn_graph = kneighbors_graph(
    pm25_data_normalized, k, mode='connectivity', include_self=True)
adjacency_matrix = torch.tensor(knn_graph.toarray(), dtype=torch.float32)

# Define a simple GCN model


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()
        self.gc1 = nn.Linear(nfeat, nhid)
        self.gc2 = nn.Linear(nhid, nclass)

    def forward(self, x, adj):
        x = torch.relu(self.gc1(torch.mm(adj, x)))
        x = self.gc2(torch.mm(adj, x))
        return x


# Parameters
nfeat = pm25_data_normalized.shape[1]
nhid = 16
nclass = 6  # You can choose the number of clusters
lr = 0.01
epochs = 200

# Convert data to torch tensor
features = torch.tensor(pm25_data_normalized, dtype=torch.float32)

# Initialize GCN model
model = GCN(nfeat, nhid, nclass)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Labels for visualization (random initialization)
labels = torch.randint(0, nclass, (features.shape[0],))

# Training the GCN model
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(features, adjacency_matrix)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Get the output embeddings
model.eval()
with torch.no_grad():
    embeddings = model(features, adjacency_matrix)

# Use t-SNE to reduce the dimensionality of the embeddings for visualization
tsne = TSNE(n_components=2)
embeddings_2d = tsne.fit_transform(embeddings.numpy())

# Plotting the clusters with labels
plt.figure(figsize=(10, 6))
for i in range(nclass):
    idx = labels == i
    plt.scatter(embeddings_2d[idx, 0],
                embeddings_2d[idx, 1], label=f'Cluster {i+1}')

# Annotating each point with its cell number
for i in range(embeddings_2d.shape[0]):
    plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1],
             str(i + 1), fontsize=9, ha='right')

plt.legend()
plt.title('GCN-based Clustering of PM2.5 Data (36 Cells)')
plt.show()
