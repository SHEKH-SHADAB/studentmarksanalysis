import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Function to display categorical feature visualizations
def visualize_categorical_features(df, cf):
    st.write("### Visualizing Categorical Features")
    n = 2
    rows = math.ceil(len(cf) / n)
    fig, axes = plt.subplots(rows, n, figsize=(10, 5 * rows))
    axes = axes.flatten()
    for i, feature in enumerate(cf):
        sns.countplot(x=df[feature], ax=axes[i])
        axes[i].set_title(f"Countplot of {feature}")
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    st.pyplot(fig)

    st.write("### Violin Plots")
    fig, axes = plt.subplots(rows, n, figsize=(10, 5 * rows))
    axes = axes.flatten()
    for i, feature in enumerate(cf):
        sns.violinplot(x=df[feature], y=df['Marks'], ax=axes[i])
        axes[i].set_title(f"Violin Plot of {feature} vs Marks")
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    st.pyplot(fig)

# Function to display numeric feature visualizations
def visualize_numeric_features(df, nf):
    st.write("### Numeric Features Distribution")
    n = 2
    rows = math.ceil(len(nf) / n)
    fig, axes = plt.subplots(rows, n, figsize=(10, 5 * rows))
    axes = axes.flatten()
    for i, feature in enumerate(nf):
        sns.histplot(df[feature], kde=True, ax=axes[i], color='blue', bins=30)
        axes[i].set_title(f"Distribution of {feature}")
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    st.pyplot(fig)

    st.write("### Boxplots")
    fig, axes = plt.subplots(rows, n, figsize=(10, 5 * rows))
    axes = axes.flatten()
    for i, feature in enumerate(nf):
        sns.boxplot(x=df[feature], ax=axes[i], color='green')
        axes[i].set_title(f"Boxplot of {feature}")
    st.pyplot(fig)

# Function to visualize pairplots
def visualize_pairplots(df):
    st.write("### Pairplot of All Features")
    pairplot_fig = sns.pairplot(df, diag_kind='kde', plot_kws={'alpha': 0.5})
    st.pyplot(pairplot_fig)

def main():
    st.sidebar.header("Upload File")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV or Excel format)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format.")
            return

        st.sidebar.write("### Uploaded Dataset Preview")
        st.sidebar.write(df.head())

        target = 'Marks'
        features = [col for col in df.columns if col != target]

        if target not in df.columns:
            st.error(f"The dataset must contain a '{target}' column.")
            return

        # Splitting features
        nu = df[features].nunique().sort_values()
        cf = [col for col in nu.index if nu[col] <= 16]
        nf = [col for col in nu.index if nu[col] > 16]

        # Visualize categorical features
        if cf:
            visualize_categorical_features(df, cf)
        else:
            st.write("No categorical features to display.")

        # Visualize numeric features
        if nf:
            visualize_numeric_features(df, nf)
        else:
            st.write("No numeric features to display.")

        # Visualize pairplots
        if len(features) > 1:
            visualize_pairplots(df)
        else:
            st.write("Not enough features for a pairplot.")
    else:
        st.write("### Upload the file to Analyze")

if __name__ == "__main__":
    main()
