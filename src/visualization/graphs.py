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

def plot_transaction_counts(nodes_df, label_of_interest='illicit'):
    # Group by time_step and class_label, count occurrences
    counts = (
        nodes_df
        .groupby(['time_step', 'class_label'])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    # Calculate the share of the selected label
    if label_of_interest not in counts.columns:
        raise ValueError(f"Label '{label_of_interest}' not found in class_label column.")
    
    label_share = counts[label_of_interest] / counts.sum(axis=1)

    # Create figure
    fig = go.Figure()

    # Add bars for each class_label
    for label in counts.columns:
        fig.add_bar(
            x=counts.index,
            y=counts[label],
            name=label
        )

    # Add selected label share as a line on a secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=label_share.index,
            y=label_share,
            mode='lines+markers',
            name=f'{label_of_interest.capitalize()} / total',
            yaxis='y2',
            line=dict(width=2, color='black')
        )
    )

    # Update layout
    fig.update_layout(
        title=f'Transaction counts and {label_of_interest} share by time step',
        xaxis_title='Time step',
        yaxis_title='Number of transactions',
        yaxis2=dict(
            title=f'{label_of_interest.capitalize()} share',
            overlaying='y',
            side='right',
            tickformat='.0%',
            rangemode='tozero'
        ),
        barmode='stack',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_white',
        margin=dict(t=70, r=80)
    )

    fig.show()
