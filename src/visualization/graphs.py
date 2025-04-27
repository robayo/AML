import pandas as pd
import networkx as nx
import plotly.graph_objects as go

def plot_transaction_graph(nodes_df, edges_df, time_step, label):
    # Filter nodes based on time_step and label
    mapper = (nodes_df['time_step'] == time_step)
    if label is not None: 
        mapper = mapper & (nodes_df['class_label'] == label)
    else:
        label = 'All'
    
    selected_ids = nodes_df[mapper]['txId']
    
    # Filter edges whose source is among the selected nodes
    selected_edges = edges_df[edges_df['src'].isin(selected_ids)]
    
    # Create directed graph from filtered edges
    graph = nx.from_pandas_edgelist(selected_edges, source='src', target='dst', create_using=nx.DiGraph())
    
    # Generate node positions
    pos = nx.spring_layout(graph)

    # Create edge traces
    edge_x, edge_y = [], []
    for src, dst in graph.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='blue'),
        hoverinfo='none',
        mode='lines'
    )

    # Create node traces
    node_x, node_y, node_text = [], [], []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_labels = nodes_df[nodes_df['txId'].isin(graph.nodes())]['class_label']
    node_colors = pd.to_numeric(node_labels.replace(['unknown', 'illicit', 'licit'], [2, 1, 3]))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_colors,
            size=10,
            line_width=2,
            colorbar=dict(
                thickness=15,
                title='Transaction Type',
                xanchor='left',
                tickmode='array',
                tickvals=[2, 1, 3],
                ticktext=['Unknown', 'Illicit', 'Licit']
            )
        )
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Transactions - {label.capitalize()} (Time Step {time_step})",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(
                showarrow=True,
                xref="paper", yref="paper",
                x=0.005, y=-0.002
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    return fig
