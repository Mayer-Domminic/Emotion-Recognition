import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

def plot_model_comparison(model_results):
    models = list(model_results.keys())
    train_scores = [results['train_acc'] for results in model_results.values()]
    test_scores = [results['test_acc'] for results in model_results.values()]
    val_scores = [results['val_acc'] for results in model_results.values()]

    fig = go.Figure(data=[
        go.Bar(name='Training', x=models, y=train_scores, marker_color='rgb(55, 83, 109)'),
        go.Bar(name='Testing', x=models, y=test_scores, marker_color='rgb(26, 118, 255)'),
        go.Bar(name='Validation', x=models, y=val_scores, marker_color='rgb(158, 202, 225)')
    ])

    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Accuracy Score',
        barmode='group',
        yaxis_range=[0, 1],
        template='plotly_white',
        hoverlabel=dict(bgcolor="white"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_nn_training(history):
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Loss over Epochs', 'Accuracy over Epochs'),
        vertical_spacing=0.15
    )

    # Plot loss
    fig.add_trace(
        go.Scatter(x=list(range(1, len(history['train_loss']) + 1)), 
                  y=history['train_loss'],
                  name='Training Loss',
                  line=dict(color='rgb(55, 83, 109)')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=list(range(1, len(history['test_loss']) + 1)), 
                  y=history['test_loss'],
                  name='Testing Loss',
                  line=dict(color='rgb(26, 118, 255)')),
        row=1, col=1
    )

    # Plot accuracy
    fig.add_trace(
        go.Scatter(x=list(range(1, len(history['train_acc']) + 1)), 
                  y=history['train_acc'],
                  name='Training Accuracy',
                  line=dict(color='rgb(55, 83, 109)')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=list(range(1, len(history['test_acc']) + 1)), 
                  y=history['test_acc'],
                  name='Testing Accuracy',
                  line=dict(color='rgb(26, 118, 255)')),
        row=2, col=1
    )

    fig.update_layout(
        height=800,
        title_text="Neural Network Training Metrics",
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=2, col=1)
    fig.update_xaxes(title_text="Epochs", row=2, col=1)

    return fig