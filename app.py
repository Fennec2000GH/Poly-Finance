import io, numpy as np, pandas as pd, streamlit as st

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

st.subheader(body='> Compare and contrast 2 datasets')
st.subheader(body='> Find common features')
st.subheader(body='> Generate new features from current features')
st.subheader(body='> Mine relationships between variables across datasets')

# upload data
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
            st.dataframe(data=st.session_state.df1)

    with col2:
        st.session_state.data2 = st.file_uploader(
            label='Upload Second CSV File',
            type='csv'
        )

        if st.session_state.data2:
            csv_str = io.StringIO(initial_value=st.session_state.data2.read().decode(encoding='utf-8'))
            st.session_state.df2 = pd.read_csv(csv_str, sep=',')
            st.dataframe(data=st.session_state.df2)

# feature generation
with st.container():
    col1, col2 = st.columns(spec=2)
    try:
        with st.container():
            st.write(f'Filter features from {st.session_state.data1.name}')
            with col1:
                for col in st.session_state.df1.columns:
                    st.checkbox(label=col)

                st.session_state.pred1 = st.selectbox(
                    label='Feature to Predict',
                    options=np.asarray(a=st.session_state.df1.columns)
                )

        with st.container():
            with col2:
                st.write(f'Filter features from {st.session_state.data2.name}')
                for col in st.session_state.df2.columns:
                    st.checkbox(label=col)

                st.session_state.pred2 = st.selectbox(
                    label='Feature to Predict',
                    options=np.asarray(a=st.session_state.df2.columns)
                )

    except:
        pass

# classification
st.selectbox(
    label='Classification Algorithm',
    options=np.asarray(a=list([
        'K Nearest Naighbors',
        'Support Vector Machine',
        'Decision Tree',
        'Random Forest'
    ]))
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
    ]))
)

if st.button(label='Regress'):
    st.balloons()
