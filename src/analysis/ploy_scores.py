import plotly.express as px
import pandas as pd


def plot_rag_scores(
    scores: pd.Series,
    title: str = "Accuracy of different RAG configurations",
):
    """
    Plot accuracy scores for different RAG configurations.
    """
    # Convert Series -> DataFrame để plotly express dùng dễ hơn
    df = scores.reset_index()
    df.columns = ["settings", "score"]

    fig = px.bar(
        df,
        x="settings",
        y="score",
        color="score",
        labels={
            "score": "Accuracy",
            "settings": "Configuration",
        },
        color_continuous_scale="bluered",
        text="score",
    )

    fig.update_layout(
        width=1000,
        height=600,
        yaxis_range=[0, 100],
        title=f"<b>{title}</b>",
        xaxis_title="RAG settings",
        font=dict(size=15),
    )

    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.layout.yaxis.ticksuffix = "%"
    fig.update_xaxes(tickangle=0)
    fig.update_coloraxes(showscale=False)

    fig.show()
