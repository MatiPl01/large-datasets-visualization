import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from sklearn.preprocessing import StandardScaler
from snn import SNN
from sklearn.cluster import KMeans

sns.set_theme()


sys.path.append("/content/")

st.set_page_config(layout="wide")

# To run Jarvis Patrick library we will use code coming from [SharedNearestNeighbors](https://github.com/felipeangelimvieira/SharedNearestNeighbors/blob/main/shared_nearest_neighbors/shared_nearest_neighbors.py) repository

def clusterize(df, n_neighbors=10, eps=5, min_samples=20, metric="euclidean"):
    snn = SNN(n_neighbors=n_neighbors, eps=eps, min_samples=min_samples, metric=metric)
    snn.fit(df)

    df["labels"] = snn.labels_
    return df

def clusterize_kmeans(df, n_clusters=8):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(df)

    df["labels"] = kmeans.labels_
    return df

# Below code defines sidebar with clustering parameters

def get_sidebar(df):
    sidebar = st.sidebar

    df_display = sidebar.checkbox("Display Raw Data", value=True)

    sidebar.title("Jarvis-Patrick clustering")

    columns = sidebar.multiselect(
        "Columns:", df.columns, max_selections=2, default=[df.columns[0], df.columns[1]]
    )

    # TODO - add selectbox with metrics (task 1)
    pairwise_distances = sidebar.selectbox('Pairwise distances', ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'])

    n_neighbors = sidebar.slider(
        "Select Number of Neighbors",
        min_value=2,
        max_value=40,
        value=7,
    )

    eps = sidebar.slider(
        "Epsilon",
        min_value=1,
        max_value=40,
        value=5,
    )

    min_samples = sidebar.slider(
        "Minimum number of samples",
        min_value=1,
        max_value=40,
        value=5,
    )

    sidebar.divider()
    sidebar.title("KMeans clustering")

    settings = {"n_neighbors": n_neighbors, "eps": eps, "min_samples": min_samples, "pairwise_distances": pairwise_distances}
    return sidebar, df_display, settings, columns


# Below functions visualizes clustering results

def plot_clusters(df, col_x="volatile acidity", col_y="pH"):
    fig, ax = plt.subplots(figsize=(16, 9))
    n_colors = df["labels"].unique().shape[0]
    print(df["labels"].unique())
    print("n_colors = ", n_colors)

    ax = sns.scatterplot(
        ax=ax,
        x=df[col_x],
        y=df[col_y],
        hue=df["labels"],
        palette=sns.color_palette("colorblind", n_colors=n_colors),
        legend=None,
    )

    return fig

def create_df(file):
    return pd.read_csv(file, sep=";")


# Below function defines page layout

def run_page():
    st.title("Interactive Clustering")

    file = st.file_uploader("Choose a file")

    if file is not None:
        df = create_df(file)
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df.values), columns=df.columns)

        sidebar, df_display, clustering_settings, columns = get_sidebar(df)

        if df_display:
            st.write(df)

        df = clusterize(
            df,
            n_neighbors=clustering_settings["n_neighbors"],
            min_samples=clustering_settings["min_samples"],
            eps=clustering_settings["eps"],
            metric=clustering_settings["pairwise_distances"],
        )
        st.write(plot_clusters(df, columns[0], columns[1]))

        st.divider()

        # TODO - add second plot with KMeans clustering (task 2)
        df = clusterize_kmeans(df, n_clusters=8)
        st.write(plot_clusters(df, columns[0], columns[1]))


run_page()
