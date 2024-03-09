import streamlit as st
import os
from openai import OpenAI
from streamlit_custom_notification_box import custom_notification_box

def show_notification(message, icon="info"):
  custom_notification_box(icon=icon, textDisplay=message, key="notification")

API_KEY = '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'
client = OpenAI(api_key=API_KEY)

def generate_itinerary(dest, src, price_per_person, num_days, num_tourists, genre, type_of_travelers, avg_age, food, mode):
    user_message = f"Design a detailed itinerary for a trip from {src} to {dest} starting from {start_date} and for " \
                   f"{num_days} days. The budget for this trip is {price_per_person} INR per person. This trip is designed " \
                   f"for {num_tourists} mainly with their {type_of_travelers} with an average age of {avg_age}.The " \
                   f"primary interests for activities are {genre}.The preferred mode(s) of travel include " \
                   f"{mode}.The group prefers {food} food. Please structure the itinerary with a detailed " \
                   f"plan for each day, including activities, locations, weather according to the season they are " \
                   f"travelling and estimated travel distances and times.Ensure to consider the preferences and " \
                   f"interests of the group for each day's schedule. Important considerations: Factor in travel time " \
                   f"between destinations. Suggest local transportation options. Include a mix of activities that cater" \
                   f" to the group's interests. Also add distance of travel for each day and approx time " \
                   f"of travel. Also you can give a name for each day in the itinerary which will be more " \
                   f"appealing. Keep the response descriptive and appealing"

    # Generate the travel itinerary using the modified user message
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_message,
            }
        ],
        model="gpt-3.5-turbo",
        stream=True,
    )

    # response_content = chat_completion.choices[0].message.content
    response = st.write_stream(chat_completion)
    return response


past_itineraries = []  # List to store past itineraries

st.title("Tour Itinerary Generator")

col1, col2 = st.columns(2)


dest = col1.text_input("Destination", key='dest')
src = col1.text_input("Source City", key='src')
genre = col1.text_input("Genre", key='genre')
type_of_travelers = col1.text_input("Type of Travelers", key='type', placeholder='ex. family, friends')
mode_of_travel = col1.text_input("Mode of Travel", key='mode', placeholder='ex. flight, bus, train')
num_days = col2.number_input("Number of Days", key='num_days')
start_date = col2.date_input("Start Date", key='start_date')
num_tourists = col2.number_input("Number of tourists", key='num_tourists')
price_per_person = col2.number_input("Price Per Person", key='price_per_person')
average_age = col2.number_input("Average age", key='average_age')
non_veg = st.toggle('Include non-veg hotels')

if non_veg:
    food = 'non veg'
else:
    food = 'veg'
if st.button("Generate Itinerary", type="primary"):
    if dest is None or src is None or price_per_person is None or num_days is None or num_tourists is None or genre is None or type_of_travelers is None or average_age is None or food or mode_of_travel is None:
        st.warning('Please enter all fields!')

    else:
        generated_itinerary = generate_itinerary(dest, src, price_per_person, num_days, num_tourists, genre, type_of_travelers,
                                                 average_age, food, mode_of_travel)
        past_itineraries.append(generated_itinerary)  # Add to past itineraries

with st.expander("Past Itineraries"):
    if past_itineraries:
        for itinerary in past_itineraries:
            st.write(itinerary)
    else:
        st.write("No past itineraries yet.")

