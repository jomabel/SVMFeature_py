import plotly.express as px
import plotly.graph_objects as go



def plot_pareto_front(df, method):
    front0 = df[df.FRONT == 0]

    config = {
        'DIST-EPS': {
            'x': 'DIST', 'y': 'EPS', 'z': None,
            'xlabel': 'DIST', 'ylabel': 'EPS', 'title': 'Pareto Front'
        },
        'MC': {
            'x': 'MC_POS', 'y': 'MC_NEG', 'z': None,
            'xlabel': 'MC_POS', 'ylabel': 'MC_NEG', 'title': 'Pareto Front'
        },
        'DIST-EPS-COST': {
            'x': 'DIST', 'y': 'EPS', 'z': 'COST',
            'xlabel': 'DIST', 'ylabel': 'EPS', 'zlabel': 'COST', 'title': '3D Pareto Front'
        },
        'MC-COST': {
            'x': 'MC_POS', 'y': 'MC_NEG', 'z': 'COST',
            'xlabel': 'MC_POS', 'ylabel': 'MC_NEG', 'zlabel': 'COST', 'title': '3D Pareto Front'
        }
    }

    cfg = config.get(method)
    if not cfg:
        raise ValueError(f"Método no reconocido: {method}")

    x, y = front0[cfg['x']], front0[cfg['y']]

    if cfg['z'] is None:
        fig = px.scatter(front0, x=cfg['x'], y=cfg['y'],
                         title=cfg['title'], labels={cfg['x']: cfg['xlabel'], cfg['y']: cfg['ylabel']})
    else:
        z = front0[cfg['z']]
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=5, opacity=0.8)
        )])
        fig.update_layout(title=cfg['title'],
                          scene=dict(
                              xaxis_title=cfg['xlabel'],
                              yaxis_title=cfg['ylabel'],
                              zaxis_title=cfg['zlabel']
                          ))

    return fig
