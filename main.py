import math

import requests
import streamlit as st
import os
from openai import OpenAI
from geopy.geocoders import Nominatim


API_KEY = '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'
client = OpenAI(api_key=API_KEY)

function_descriptions = [
    {
        "name": "get_flight_hotel_info",
        "description": "Find the flights between cities and hotels within cities for residence.",
        "parameters": {
            "type": "object",
            "properties": {
                "loc_list": {
                  "type": "array",
                  "items": {
                      "type": "string"
                  },
                  "description": "The ordered list of IATA codes of cities in the tour. e.g. ['BOM', 'PNQ']"
              },

                "date_list": {
                  "type": "array",
                  "items": {
                      "type": "string"
                  },
                  "description": "The ordered list of dates for arrival in the cities in YYYY-MM-DD format."
              },

            },
            "required": ["loc_list", "date_list"],
        },
    }
]

def get_hotels(city, checkin_date, checkout_date, num_adults, num_children, city_dict):
    # Initialize Nominatim API
    st.write(city_dict)
    geolocator = Nominatim(user_agent="MyApp")

    location = geolocator.geocode(city)
    lat = location.latitude
    long = location.longitude
    print("The latitude of the location is: ", location.latitude)
    print("The longitude of the location is: ", location.longitude)

    url = "https://booking-com.p.rapidapi.com/v1/hotels/search-by-coordinates"

    num_rooms = math.ceil((num_adults + num_children) / 3)
    querystring = {
        "locale": "en-gb",
        "room_number": num_rooms,
        "checkin_date": checkin_date,
        "checkout_date": checkout_date,
        "filter_by_currency": "INR",
        "longitude": long,
        "latitude": lat,
        "adults_number": num_adults,
        "order_by": "popularity",
        "units": "metric",
        "page_number": "0",
        "children_number": num_children,
        "children_ages": "5,5,5,5,5",
        "include_adjacency": "true",
        "categories_filter_ids": "class::2,class::4,free_cancellation::1"
    }

    headers = {
        "X-RapidAPI-Key": "8d111f0846msha3a0a23cb3dce84p149309jsndff256e053ad",
        "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    st.write(response.status_code)
    if response.status_code == 200:
        data = response.json()
        if "result" in data:
            city_dict[city] = []
            print("Hotel search results for Mumbai:")
            for index, hotel in enumerate(data["result"], start=1):
                hotel_dict = {}
                hotel_dict['hotel_name'] = hotel.get("hotel_name", "N/A")
                price = hotel.get("min_total_price", "N/A")
                if isinstance(price, int):
                    hotel_dict['price'] = str(price)
                hotel_dict['address'] = hotel.get("address", "N/A")
                hotel_dict['rating'] = hotel.get("review_score", "N/A")
                # print(f"{index}. {hotel_name} - Address: {address}, Price for one day: {price} AED, Rating: {rating}")
                st.write(hotel_dict)
                city_dict[city].append(hotel_dict)
        else:
            print("No hotel results found.")
    else:
        print("Failed to retrieve hotel search results.")


def generate_itinerary(input_dict):
    # Part 1: generate the list of cities and get the hotels
    # Call the OpenAI API for creating the list of cities and dates
    city_dict = {}
    cities = ['Pune', 'Mumbai', 'Hyderabad']
    dates = ['2024-04-01', '2024-04-03', '2024-04-05', '2024-04-07']
    for i in range(len(cities)):
        get_hotels(cities[i], dates[i], dates[i+1], input_dict['num_adults'], input_dict['num_children'], city_dict)

    # Part 2: Actually generate the itinerary
    user_message = f"Design a detailed itinerary for a trip from {input_dict['src']} to {input_dict['dest']} starting from {input_dict['start_date']} and for " \
                   f"{input_dict['num_days']} days. The budget for this trip is {input_dict['price_per_person']} INR per person. This trip is designed " \
                   f"for {input_dict['num_tourists']} mainly with their {input_dict['type_of_travelers']} with an average age of {input_dict['average_age']}.The " \
                   f"primary interests for activities are {input_dict['genre']}.The preferred mode(s) of travel include " \
                   f"{input_dict['mode_of_travel']}.The group prefers {input_dict['food']} food. Please structure the itinerary with a detailed " \
                   f"plan for each day, including activities, locations, weather according to the season they are " \
                   f"travelling and estimated travel distances and times.Ensure to consider the preferences and " \
                   f"interests of the group for each day's schedule. Important considerations: Factor in travel time " \
                   f"between destinations. Suggest local transportation options. Include a mix of activities that cater" \
                   f" to the group's interests. Also add distance of travel for each day and approx time " \
                   f"of travel. Also you can give a name for each day in the itinerary which will be more " \
                   f"appealing. Keep the response descriptive and appealing"

    # Generate the travel itinerary using the modified user message
    # chat_completion = client.chat.completions.create(
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": user_message,
    #         }
    #     ],
    #     model="gpt-3.5-turbo",
    #     stream=True,
    # )
    #
    # # response_content = chat_completion.choices[0].message.content
    # response = st.write_stream(chat_completion)
    response = "Dummy response"
    return response, city_dict


past_itineraries = []  # List to store past itineraries
input_dict = {}
st.title("Tour Itinerary Generator")

col1, col2 = st.columns(2)

input_dict['dest'] = col1.text_input("Destination", key='dest')
input_dict['src'] = col1.text_input("Source City", key='src')
input_dict['genre'] = col1.text_input("Genre", key='genre')
input_dict['type_of_travelers'] = col1.text_input("Type of Travelers", key='type', placeholder='ex. family, friends')
input_dict['mode_of_travel'] = col1.text_input("Mode of Travel", key='mode', placeholder='ex. flight, bus, train')
input_dict['num_days'] = col2.number_input("Number of Days", key='num_days')
input_dict['start_date'] = col2.date_input("Start Date", key='start_date')
# Create sub-columns within col2
col21, col22 = col2.columns(2)

input_dict['num_adults'] = col21.number_input("Number of Adults", key='num_adults')
input_dict['num_children'] = col22.number_input("Number of Children", key='num_children')
input_dict['price_per_person'] = col2.number_input("Price Per Person", key='price_per_person')
input_dict['average_age'] = col2.number_input("Average age", key='average_age')
input_dict['food'] = 'non veg' if st.toggle('Include non-veg hotels') else 'veg'

input_dict['num_tourists'] = input_dict['num_adults'] + input_dict['num_children']

if st.button("Generate Itinerary", type="primary"):
    for key in input_dict.keys():
        if input_dict[key] is None:
            st.warning(f'Please enter {key}!')

    else:
        generated_itinerary, city_dict = generate_itinerary(input_dict)
        past_itineraries.append(generated_itinerary)  # Add to past itineraries
        print(city_dict.items())
        st.write(city_dict)
with st.expander("Past Itineraries"):
    if past_itineraries:
        for itinerary in past_itineraries:
            st.write(itinerary)
    else:
        st.write("No past itineraries yet.")

