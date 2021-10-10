import io, numpy as np, pandas as pd, streamlit as st
import plotly.express as px

from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext, dataframe

from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

# setting up PySpark
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.master(master='local').appName(name='Poly-Finance').getOrCreate()

# logo and badges
st.set_page_config(
    page_title='Poly Finance',
    page_icon='💸',
    layout='centered',
    initial_sidebar_state='expanded'
)

st.image(image='./logo.gif', use_column_width=True, clamp=True)

badges = list(['![language](https://img.shields.io/badge/language-python-yellow?style=plastic&logo=appveyor)',
'![backend](https://img.shields.io/badge/framework-Apache%20Spark-orange?style=for-the-badge&logo=appveyor)',
'[![Star](https://img.shields.io/github/stars/Fennec2000GH/Poly-Finance.svg?logo=github&style=social)](https://gitHub.com/Fennec2000GH/IntelliVision)'])

st.write((' ' * 5).join(badges))

# introductory headers
st.title(body='Poly Finance')
st.header(body='Where Big Data Meets Finance')

st.write('> 🎯 Compare and contrast 2 datasets')
st.write('> 🎯 Select custom features for visualization')
st.write('> 🎯 Generate new features from current features')
st.write('> 🎯 Compute descriptive statistics')
st.write('> 🎯 Perform PCA analysis')
st.write('> 🎯 Run machine learning algorithms to evaluate data')
st.session_state.valid = False

# upload data
st.subheader(body='Upload Data')
with st.container():
    col1, col2 = st.columns(spec=2)

    with col1:
        st.session_state.data1 = st.file_uploader(
            label='Upload First CSV File',
            type='csv'
        )

        if st.session_state.data1:
            csv_str = io.StringIO(initial_value=st.session_state.data1.read().decode(encoding='utf-8'))
            st.session_state.df1 = pd.read_csv(csv_str, sep=',')
            st.session_state.df1_spark = spark.createDataFrame(data=st.session_state.df1)
            st.dataframe(data=st.session_state.df1)
            
    with col2:
        st.session_state.data2 = st.file_uploader(
            label='Upload Second CSV File',
            type='csv'
        )

        if st.session_state.data2:
            csv_str = io.StringIO(initial_value=st.session_state.data2.read().decode(encoding='utf-8'))
            st.session_state.df2 = pd.read_csv(csv_str, sep=',')
            st.session_state.df2_spark = spark.createDataFrame(data=st.session_state.df2)
            st.dataframe(data=st.session_state.df2)

    if st.session_state.data1 and st.session_state.data2:
        st.session_state.valid = True

# feature selection
st.subheader(body='Feature Selection')
with st.container():
    col1, col2 = st.columns(spec=2)
    col1_filtered, col2_filtered = st.columns(spec=2)
    st.session_state.checkboxes1 = dict()
    st.session_state.checkboxes2 = dict()
    
    if st.session_state.valid:
        st.session_state.valid = False

        with col1:
            for col in st.session_state.df1.columns:
                st.session_state.checkboxes1[col] = st.checkbox(label=col, key=f'{col}_1')

            st.session_state.pred1 = st.selectbox(
                label='Feature to Predict',
                options=np.asarray(a=st.session_state.df1.columns),
                help='Class or label for color differentiation during plotting.'
            )

        with col1_filtered:
            filtered_cols = list([key for key, value in list(st.session_state.checkboxes1.items()) if value])
            st.session_state.df1_filtered = st.session_state.df1_spark.select(filtered_cols).toPandas()
            st.dataframe(data=st.session_state.df1_filtered)

        with col2:
            for col in st.session_state.df2.columns:
                st.session_state.checkboxes2[col] = st.checkbox(label=col, key=f'{col}_2')

            st.session_state.pred2 = st.selectbox(
                label='Feature to Predict',
                options=np.asarray(a=st.session_state.df2.columns),
                help='Class or label for color differentiation during plotting.'
            )

        with col2_filtered:
            filtered_cols = list([key for key, value in list(st.session_state.checkboxes2.items()) if value])
            st.session_state.df2_filtered = st.session_state.df2_spark.select(filtered_cols).toPandas()
            st.dataframe(data=st.session_state.df2_filtered)

        st.session_state.valid = True

# visualization
st.subheader(body='Visualization')
with st.container():
    col1, col2 = st.columns(spec=2)

    if st.session_state.valid:
        # st.session_state.valid = False

        with col1:
            fig = px.scatter_matrix(
                data_frame=st.session_state.df1,
                dimensions=st.session_state.df1_filtered.columns,
                color=st.session_state.pred1
            )

            fig.update_traces(diagonal_visible=False)

            st.plotly_chart(
                figure_or_data=fig,
                use_container_width=True
            )

        with col2:
            fig = px.scatter_matrix(
                data_frame=st.session_state.df2,
                dimensions=st.session_state.df2_filtered.columns,
                color=st.session_state.pred2
            )

            fig.update_traces(diagonal_visible=False)

            st.plotly_chart(
                figure_or_data=fig,
                use_container_width=True
            )

    st.session_state.valid = True

# merge datasets
st.subheader(body='Merge Datasets')
with st.container():
    if st.session_state.valid:
        st.session_state.valid = False
        st.session_state.df_merged = st.session_state.df1_filtered.join(other=st.session_state.df2_filtered)
        st.dataframe(data=st.session_state.df_merged)

    st.session_state.valid = True

# feature generation
st.subheader(body='Polynomial Feature Generation')
with st.container():

    if st.session_state.valid:
        st.session_state.valid = False
        degree = st.number_input(
            label='Degree',
            min_value=1,
            max_value=5,
            step=1,
            help='Maximal degree of features generated.'
        )

        interaction_only = st.checkbox(
            label='Interaction Only',
            help="""
            If True, then only way to generate higher-degree features is by multiplying different existing features together.
            Otherwise, features are generated by raising individual existing features to a higher degree on their own.
            """
        )

        poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only
        )

        st.write(poly.fit_transform(X=st.session_state.df_merged))

    st.session_state.valid = True

# descriptive statistics
st.subheader(body='Descriptive Statistics')
with st.container():

    if st.session_state.valid:
        st.session_state.valid = False
        st.dataframe(data=st.session_state.df_merged.describe())

    st.session_state.valid = True

# PCA
if st.checkbox(label='PCA'):
    st.subheader(body='Principal Component Analysis')
    pca = PCA()
    pca.fit(X=st.session_state.df_merged)
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

    pca_explained_area = px.area(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        labels={"x": "# Components", "y": "Explained Variance"}
    )

    st.plotly_chart(
        figure_or_data=pca_explained_area,
        use_container_width=True
    )


# machine learning
st.subheader(body='Machine Learning')
with st.container():

    if st.session_state.valid:
        st.session_state.valid = False

        # classification
        st.selectbox(
            label='Classification Algorithm',
            options=np.asarray(a=list([
                'K Nearest Naighbors',
                'Support Vector Machine',
                'Decision Tree',
                'Random Forest'
            ])),
            help='Choose algorithm for classification.'
        )

        if st.button(label='Classify'):
            st.balloons()

        # regression
        st.selectbox(
            label='Regression Algorithm',
            options=np.asarray(a=list([
                'K Nearest Naighbors',
                'Support Vector Machine',
                'Decision Tree',
                'Random Forest'
            ])),
            help='Choose algorithm for regression.'
        )

        if st.button(label='Regress'):
            st.balloons()

    st.session_state.valid = True


