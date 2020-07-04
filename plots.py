import plotly.graph_objects as go


def plot_3d(dataframe_X, y):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(x=dataframe_X['pca_one'], y=dataframe_X['pca_two'], z=dataframe_X['pca_three'], mode='markers',
                     marker=dict(
                         size=10,
                         color=y,
                         colorscale='Viridis',
                         opacity=0.8
                     )))
    fig.show()
