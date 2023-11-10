# Import necessary libraries
import streamlit as st
import pickle
from PIL import Image

# Load the trained model and create a function to make predictions
with open('rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def predict_class(sepal_length, sepal_width, petal_length, petal_width):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)
    return prediction[0]

# Set page title and favicon
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌼",
)

# Streamlit app header
st.title("IRIS FLOWER CLASSIFIER")

# Sidebar for user input
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 0.0, 8.0, 0.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 0.0, 5.0, 0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", 0.0, 7.5, 0.1)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.0, 3.0, 0.1)

# Get the prediction
prediction = predict_class(sepal_length, sepal_width, petal_length, petal_width)

# Iris class labels
iris_classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

# Display the predicted Iris class with a fancy emoji
st.write("Predicted Iris Class:", f"🌸 {prediction} 🌸")

# Display the corresponding image based on the prediction with some padding and captions
if prediction == iris_classes[0]:
    st.image("Iris-setosa.jpeg", use_column_width=True)
elif prediction == iris_classes[1]:
    st.image("Iris-versicolor.jpeg", use_column_width=True)
elif prediction == iris_classes[2]:
    st.image("Iris-virginica.jpeg", use_column_width=True)

# Add an explanation for the prediction
st.subheader("Iris Species Explanation:")
if prediction == iris_classes[0]:
    st.write("Iris Setosa is a distinct and elegant species known for its striking appearance. It has narrow sepals, short petals, and comes in various colors.")
elif prediction == iris_classes[1]:
    st.write("Iris Versicolor is a versatile species with a wide range of colors and patterns. It is admired for its medium-sized petals and sepals.")
elif prediction == iris_classes[2]:
    st.write("Iris Virginica is a vibrant and graceful species that features long, slender petals and sepals. It is highly valued for its elegant look and sweet fragrance.")