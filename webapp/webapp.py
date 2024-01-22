import streamlit as st
import pickle
import numpy as np


img1 = "https://i.ibb.co/9rq5Zfm/logo.png"
img2 = "https://i.ibb.co/HHVtJCx/Screenshot-2023-11-26-at-21-45-54-ANNONCE-SIDI-22-TA-1-5-pdf-removebg-preview.png"
st. set_page_config(layout="wide")
col11, col22 = st.columns(2)
with col11:
    st.image(img1,width=460)
    
with col22:
    st.image(img2,width=300)

st.header("Books recommender using machine learning")

model = pickle.load(open('model.pkl','rb'))
books_name = pickle.load(open('books_name.pkl','rb'))
final_rating = pickle.load(open('final_rating.pkl','rb'))
book_pivot = pickle.load(open('book_pivot.pkl','rb'))

def fetch_poster(suggestion):
    books_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        books_name.append(book_pivot.index[book_id])
    for book in books_name[0]:
        ids = np.where(final_rating['title'] == book)[0][0]
        ids_index.append(ids)
    for idx in ids_index:
        url = final_rating.iloc[idx]['img_url']
        poster_url.append(url)
    
    return poster_url

def recommend_books(book_name):
    book_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6)

    poster_url = fetch_poster(suggestion)
    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for book in books:
            book_list.append(book)

    return book_list, poster_url

selected_books = st.selectbox(
    "Type or Select a book",
    books_name
)
if st.button('Show Recommendation'):
    recommendation_books, poster_url = recommend_books(selected_books)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(recommendation_books[1])#[0] is the same book so we will ignore it
        st.image(poster_url[1])
    
    with col2:
        st.text(recommendation_books[2])
        st.image(poster_url[2])
    
    with col3:
        st.text(recommendation_books[3])
        st.image(poster_url[3])

    with col4:
        st.text(recommendation_books[4])
        st.image(poster_url[4])

    with col5:
        st.text(recommendation_books[5])
        st.image(poster_url[5])


