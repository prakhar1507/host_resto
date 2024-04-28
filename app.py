import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os 
from dotenv import load_dotenv
from PIL import Image as PILImage
import google.api_core.exceptions

# Load the saved TF-IDF model
with open('tfidf_model.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Load dataset
data = pd.read_csv('restaurants_data.csv')
df = pd.DataFrame(data)

df['cui_textual_features'] = df['Cuisine types']  + df['what_people_like_about_us'] + df['Price'] +df['Rating']

tfidf_matrix = tfidf_vectorizer.fit_transform(df['cui_textual_features'])


def recommend_cuisine(user_food, user_location, top_n=5):
    
    user_input = tfidf_vectorizer.transform([user_food])

    
    filtered_restaurants = df[df['Location'] == user_location]

    
    similarities = cosine_similarity(user_input, tfidf_matrix[filtered_restaurants.index])

    
    top_indices = similarities.argsort()[0][-top_n:][::-1]
    recommendation = filtered_restaurants.iloc[top_indices]

    return recommendation[['Restaurent name', 'Cuisine types', 'Rating', 'Price', 'Distance', 'Location', 'what_people_like_about_us']]





# Combine relevant text columns
df['textual_features'] = df['Cuisine types']  + df['what_people_like_about_us'] + df['Price'] + df['Rating']

# Apply TF-IDF to combined text column
tfidf_matrix = tfidf_vectorizer.transform(df['textual_features'])

def recommend_restaurants(restaurant_name, user_location, top_n=3):

    restaurant_row = df[df['Restaurent name'] == restaurant_name]
    # Transform user input into TF-IDF vector
    user_input = tfidf_vectorizer.transform(restaurant_row['textual_features'])

    # Filter restaurants based on user location
    filtered_restaurants = df[df['Location'] == user_location]

    # Calculate cosine similarity between user input and restaurant TF-IDF vectors
    similarities = cosine_similarity(user_input, tfidf_matrix[filtered_restaurants.index])

    # Get top N similar restaurants
    top_indices = similarities.argsort()[0][-top_n:][::-1]
    recommendations = filtered_restaurants.iloc[top_indices]

    return recommendations[['Restaurent name', 'Cuisine types', 'Rating', 'Price', 'Distance', 'Location', 'what_people_like_about_us']]

# Streamlit UI
# def main():


genai.configure(api_key="AIzaSyArKdHLVe7TjPrqsSjBa086mxD3UP7Hg8Y")


def get_gemini_response(input_prompt, image):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([input_prompt, image[0]])
    return response.text


def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")



    





tab1, tab2,tab3,tab4= st.tabs(['Restaurants', 'Cuisine','Cuisine reviews','Analysis City Wise Distribution'])

with tab1:
    st.title("Restaurant Recommendation System")

    # Dropdown for cuisine types with search functionality
    user_rest = st.selectbox("Enter preferred Restaurent:", df['Restaurent name'].unique(), key="tab1_cuisine")
    # Dropdown for locations
    user_location = st.selectbox("Enter location:", df['Location'].unique(), key="tab1_location")

    if st.button("Recommend"):
        recommendations = recommend_restaurants(user_rest, user_location)
        st.subheader("Recommendations based on your preferences:")
        st.table(recommendations)

with tab2:
    st.title("Restaurant Recommendation System - by Cuisine")

    # Dropdown for cuisine types with search functionality
    user_food = st.selectbox("Enter preferred cuisine type:", df['Cuisine types'].unique(), key="tab2_cuisine")
    # Dropdown for locations
    user_locations = st.selectbox("Enter location:", df['Location'].unique(), key="tab2_location")

    if st.button("Recommends"):
        recommendations = recommend_cuisine(user_food, user_locations)
        st.subheader("Recommendations based on your preferences:")
        st.table(recommendations)

with tab3:
    # st.set_page_config(page_title="Nutrition from Food", page_icon=":apple:")

    st.title("Nutrition from Food")

    uploaded_file = st.file_uploader("Upload your food image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        food_image = PILImage.open(uploaded_file)
        st.image(food_image, caption="Uploaded Image of Food", use_column_width=True)
        submit = st.button("Tell me about Food Nutrition")

        if submit:
            try:
                input_prompt = """
                I have given you a Food dish tell me the following accurately
                Name of Dish: [Input the name of the dish here]

Description of Dish: [Provide any additional details or ingredients if necessary]

Portion Size: [Specify the portion size, e.g., 100 grams or 100 ml if it's a drink]

Healthiness Assessment:

Is the dish generally considered healthy? [Yes/No/Neutral]

Nutritional Values (per specified portion size):

Calories: [Provide the number of calories per portion]

Fat: [Specify the amount of fat in grams per portion]

Carbohydrates: [Specify the amount of carbohydrates in grams per portion]

Protein: [If applicable, specify the amount of protein in grams per portion]

Sugar: [If applicable, specify the amount of sugar in grams per portion]

Fiber: [If applicable, specify the amount of fiber in grams per portion]

Sodium: [If applicable, specify the amount of sodium in milligrams per portion]

Other Key Nutrients: [Any other important nutrients present in significant amounts]

Additional Notes:

if the given image is not in category of food  or not related to any of the food item response me "This is not the image of Food"

[Provide any further insights or information regarding the dish's nutritional value, ingredients, or health implications.]
                """
                image_data = input_image_setup(uploaded_file)
                response = get_gemini_response(input_prompt, image_data)
                st.header("The Response is")
                st.write(response)
            except google.api_core.exceptions.InvalidArgument:
                st.error("Please upload an image of food or drink.")

with tab4:

    st.title("RESTAURANT ANALYSIS")
    st.image("p1.jpg", caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")

    st.markdown("""
    <div style="max-width: 800px; text-align: left;">
        Above Bar graph shows the average cuisine ratings of three different cities
        : Agra, Indore, and Lucknow. Lucknow has the highest average cuisine rating,
        at 3.8885. Indore follows closely behind, with an average rating of 3.8470. 
        Agra has the lowest average cuisine rating of the three cities, at 3.7866.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")  # Add a gap between images
    st.markdown("---")  # Add a gap between images

    st.image("p2.jpg", caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")
    st.markdown("""
    <div style="max-width: 800px; text-align: left;">
        Pie chart  showing information about two different aspects of these cities:
        cuisine rating and cuisine price.
        Cuisine Rating: The graph likely depicts average ratings (out of 5) given by
        reviewers for the overall dining experience in each city. Lucknow has the highest
        average rating (3.8885), followed by Indore (3.8470) and then Agra (3.7866). This
        suggests that Lucknow receives the most favorable reviews for its food scene.
        Cuisine Price: The text you mentioned likely refers to separate data and indicates
        the average cost per person for a meal (₹) in each city. Here, Agra appears to be 
        the most affordable option with an average price of ₹250.14, followed by Lucknow 
        (₹278.42) and Indore (₹321.89).It's interesting to see that while Lucknow boasts 
        the highest rated cuisine, Agra might be a more budget-friendly choice for dining.
        Indore falls somewhere in the middle for both rating and price.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")  # Add a gap between images
    st.markdown("---")  # Add a gap between images
    
    st.image("p3.jpg", caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")

    st.markdown("""
    <div style="max-width: 800px; text-align: left;">
                The bar graph  shows the average distance in kilometers at which 
                food is available in three Indian cities: Agra, Indore, and Lucknow. Lucknow
                has the shortest average distance at which food is available, at 3.926 
                kilometers. Indore follows closely behind, with an average distance of 5.391 
                kilometers. Agra has the farthest average distance at which food is available,
                at 6.0 kilometers
        
    </div>
    """, unsafe_allow_html=True)


    st.markdown("---")  # Add a gap between images
    st.markdown("---")  # Add a gap between images
    st.image("p4.jpg", caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")
    st.markdown("""
    <div style="max-width: 800px; text-align: left;">
                The graph shows the distribution of restaurants across various sub-locations 
                within three Indian cities. There are more restaurants in some sub-locations than
                others. For example, Civil Lines in Agra has the most restaurants. Lucknow appears
                to have the most restaurants out of the three cities listed.Indore showcases this 
                variety as well. 
                
        
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")  # Add a gap between images
    st.markdown("---")  # Add a gap between images
    st.image("p5.jpg", caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")

    st.markdown("""
    <div style="max-width: 800px; text-align: left;">
                “Average ratings with each sublocation of city”. The X-axis shows the city and 
                sublocations, and the Y-axis shows the average rating.The graph displays average 
                ratings for places, presumably restaurants, across various sublocations within four 
                Indian cities: Agra, Indore, Kanpur, and Lucknow.Here’s a breakdown of some of the
                places mentioned:
                Agra:
                The sublocation with the highest average rating is Dayal Bagh at 4.80.Other sublocations
                include Tajganj, Lohamandi, and Ranjeet Hanuman.
                Indore: It has the highest average rating among the four cities at 2.65.Sublocations 
                aren't mentioned for Indore.
                Lucknow:
                The sublocation with the highest average rating is Charbagh at 4.80.Other sublocations 
                include Alambagh, Gomti Nagar, and Hazratganj.
                 
                
        
    </div>
    """, unsafe_allow_html=True)


    st.markdown("---")  # Add a gap between images
    st.markdown("---")  # Add a gap between images
    st.image("p6.jpg", caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")
    st.markdown("""
    <div style="max-width: 800px; text-align: left;">
                The graph shows the average cost of a meal at various locations in three Indian cities: Agra,
                Indore, and Lucknow. Prices vary depending on the location.  Generally, restaurants in Agra 
                tend to be more expensive than restaurants in Lucknow or Indore. For example, the most expensive
                meal in Agra costs ₹1,425, while the most expensive meal in Lucknow costs ₹2,050. Sapru Marg in 
                Lucknow is the least expensive place to eat out of the ones listed, where the average meal costs 
                ₹100.
                
                 
                
        
    </div>
    """, unsafe_allow_html=True)






# if __name__ == "__main__":
#     main()
