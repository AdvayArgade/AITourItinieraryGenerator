from datetime import timedelta
import json
import math
import requests
import streamlit as st
from openai import OpenAI
from geopy.geocoders import Nominatim
import logging
from dotenv import load_dotenv
import os

load_dotenv('key.env')
API_KEY = os.getenv('OPENAI_API_KEY')

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
                  "description": "The ordered list of names of cities in the tour. e.g. ['Mumbai', 'Paris']"
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

# Set the logging level (DEBUG for most details)
logging.basicConfig(level=logging.DEBUG)

# Create a logger for your specific API usage
logger = logging.getLogger('booking_com_api')

def make_booking_com_api_call(url, headers, data):
  logger.debug("API Request:")
  logger.debug(f"URL: {url}")
  logger.debug(f"Headers: {headers}")
  logger.debug(f"Body: {data}")

  # Make your API call using requests library
  response = requests.get(url, headers=headers, json=data)

  logger.debug("API Response:")
  logger.debug(f"Status Code: {response.status_code}")
  logger.debug(f"Headers: {response.headers}")
  logger.debug(f"Response Body: {response.json()}")

  return response

def get_hotel_data(city, checkin_date, checkout_date, num_adults, num_children):
    city_dict = {}

    # Geocode city to get latitude and longitude
    geolocator = Nominatim(user_agent="MyApp")
    location = geolocator.geocode(city)
    if location:
        lat = location.latitude
        long = location.longitude

        # Define URL and query parameters for Booking.com API
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

        # Send request to Booking.com API
        response = requests.get(url, headers=headers, params=querystring)
        if response.status_code == 200:
            data = response.json()
            if "result" in data:
                city_dict[city] = []
                st.write("Hotel search results for", city, ":")
                for index, hotel in enumerate(data["result"], start=1):
                    hotel_dict = {}
                    hotel_dict['hotel_name'] = hotel.get("hotel_name", "N/A")
                    price = hotel.get("min_total_price", "N/A")
                    if price is not None:
                        hotel_dict['price'] = str(price)
                    else:
                        hotel_dict['price'] = "N/A"
                    hotel_dict['address'] = hotel.get("address", "N/A")
                    hotel_dict['rating'] = hotel.get("review_score", "N/A")
                    st.write(f"{index}. {hotel_dict['hotel_name']} - Address: {hotel_dict['address']}, Price for one day: {hotel_dict['price']} AED, Rating: {hotel_dict['rating']}")
                    city_dict[city].append(hotel_dict)
            else:
                st.write(f"No hotel results found for {city}.")
        else:
            st.write(f"Failed to retrieve hotel search results for {city}.")
    else:
        st.write(f"Invalid city name: {city}")

    return city_dict


def generate_itinerary(input_dict):
    # Part 1: generate the list of cities and get the hotels
    # Call the OpenAI API for creating the list of cities and dates
    input_dict['end_date'] = str(input_dict['start_date'] + timedelta(days=input_dict['num_days']))
    user_prompt = f"Generate a list of cities for a tour of {input_dict['dest']} for {input_dict['num_tourists']} " \
                  f"people with {input_dict['num_adults']} adults, purpose as {input_dict['genre']} " \
                  f"for {input_dict['num_days']} days and a budget per person of {input_dict['price_per_person']} INR starting on " \
                  f"{input_dict['start_date']}, ending on {input_dict['end_date']}. Call the function 'get_flight_hotel_info'"

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_prompt}],
        # Add function calling
        functions=function_descriptions,
        function_call="auto",  # specify the function call
        max_tokens=200
    )
    output = completion.choices[0].message
    cities = json.loads(output.function_call.arguments).get("loc_list")
    dates = json.loads(output.function_call.arguments).get("date_list")
    dates.append(input_dict['end_date'])

    all_city_dict = {}
    # cities = ['Pune', 'Mumbai', 'Hyderabad']
    # dates = ['2024-05-19', '2024-05-21', '2024-05-23', '2024-05-25']
    for i in range(len(cities)):
        st.write(cities[i], dates[i], dates[i+1], input_dict['num_adults'], input_dict['num_children'])
        all_city_dict.update(get_hotel_data(cities[i], dates[i], dates[i+1], input_dict['num_adults'], input_dict['num_children']))

    # Part 2: Actually generate the itinerary
    user_message = f"Include the same cities and dates from your previous response. Design a detailed itinerary for a trip from {input_dict['src']} to {input_dict['dest']} starting from {input_dict['start_date']} and for " \
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
    return response, all_city_dict

st.cache(suppress_st_warning=True)(get_hotel_data)
st.cache(suppress_st_warning=True)(generate_itinerary)
past_itineraries = []  # List to store past itineraries
input_dict = {}
st.set_page_config(
    page_title="AI Tour Itinerary Generator",  # Set your desired title here
    page_icon="images/favicon.ico",  # Set path to your favicon image (.ico format)
)
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

input_dict['num_adults'] = int(col21.number_input("Number of Adults", key='num_adults'))
input_dict['num_children'] = int(col22.number_input("Number of Children", key='num_children'))
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

with st.expander("Past Itineraries"):
    if past_itineraries:
        for itinerary in past_itineraries:
            st.write(itinerary)
    else:
        st.write("No past itineraries yet.")

