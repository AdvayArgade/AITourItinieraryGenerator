from datetime import timedelta
import json
import math
import requests
import streamlit as st
from geopy.geocoders import Nominatim
from openai import OpenAI
import logging
from dotenv import load_dotenv
import os
from docx import Document
from docx.shared import Pt
from docx.shared import Inches
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from io import BytesIO
import nltk
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from amadeus import Client, ResponseError, Location
import zipfile


st.session_state['data_changed'] = False
input_dict = {}
st.set_page_config(
    page_title="AI Tour Itinerary Generator",  # Set your desired title here
    page_icon="images/favicon.ico",  # Set path to your favicon image (.ico format)
)
st.title("Tour Itinerary Generator")

# Input fields for API keys
st.subheader("API Key Input")
API_KEY = st.text_input("Enter OpenAI API Key:")
RAPID_API_KEY = st.text_input("Enter RapidAPI Key:")
RAPID_API_HOST = st.text_input("Enter RapidAPI Host:")
if st.button("Submit API Keys"):
    # Store the API keys in session state
    st.session_state["OPENAI_API_KEY"] = API_KEY
    st.session_state["RAPID_API_KEY"] = RAPID_API_KEY
    st.session_state["RAPID_API_HOST"] = RAPID_API_HOST

    st.success("API Keys submitted successfully!")

col1, col2 = st.columns(2)

input_dict['dest'] = col1.text_input("Destination", key='dest')
input_dict['src'] = col1.text_input("Source City", key='src')
input_dict['genre'] = col1.text_input("Genre", key='genre')
input_dict['type_of_travelers'] = col1.text_input("Type of Travelers", key='type', placeholder='ex. family, friends')
input_dict['mode_of_travel'] = col1.text_input("Mode of Travel", key='mode', placeholder='ex. flight, bus, train')
input_dict['num_days'] = col2.number_input("Number of Days", key='num_days', min_value=0, max_value=None, value=0,
                                           step=1, format="%d")
input_dict['start_date'] = col2.date_input("Start Date", key='start_date')
# Create sub-columns within col2
col21, col22 = col2.columns(2)

input_dict['num_adults'] = int(
    col21.number_input("Number of Adults", key='num_adults', min_value=0, max_value=None, value=0, step=1, format="%d"))
input_dict['num_children'] = int(
    col22.number_input("Number of Children", key='num_children', min_value=0, max_value=None, value=0, step=1,
                       format="%d"))
input_dict['price_per_person'] = col2.number_input("Price Per Person", key='price_per_person', min_value=0.0)
input_dict['average_age'] = col2.number_input("Average age", key='average_age', min_value=0, max_value=None, value=0,
                                              step=1, format="%d")
input_dict['food'] = 'non veg' if st.toggle('Include non-veg hotels') else 'veg'
special_note = st.text_area("Special Note(Optional)", key='special_note')


input_dict['num_tourists'] = input_dict['num_adults'] + input_dict['num_children']


client = OpenAI(api_key=API_KEY)

amadeus = Client(
    client_id='HlpBStOyyZ79qlc8cD4dTJEnsjnBv59Z',
    client_secret='VgElo0vAcc2QLiZ5'
)

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
        }
        if num_children > 0:
            querystring["children_number"] = num_children
        headers = {
            "X-RapidAPI-Key": RAPID_API_KEY,
            "X-RapidAPI-Host": RAPID_API_HOST
        }

        # Send request to Booking.com API
        response = requests.get(url, headers=headers, params=querystring)
        if response.status_code == 200:
            data = response.json()
            if "result" in data:
                city_dict[city] = []
                st.write(f"Finding hotels for {city}")
                # st.write("Hotel search results for", city, ":")
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
                    # st.write(f"{index}. {hotel_dict['hotel_name']} - Address: {hotel_dict['address']}, Price for one day: {hotel_dict['price']} INR, Rating: {hotel_dict['rating']}")
                    city_dict[city].append(hotel_dict)
            else:
                st.write(f"No hotel results found for {city}.")
        else:
            st.write(f"Failed to retrieve hotel search results for {city}.")
    else:
        st.write(f"Invalid city name: {city}")

    return city_dict


def flight_search(input_dict):
    num_adults = input_dict['num_adults']
    cities = input_dict['cities']
    dates = input_dict['dates']
    flights = {}

    # Convert city names to IATA codes
    iata_codes = {}
    for city in cities:
        try:
            response = amadeus.reference_data.locations.get(
                keyword=city,
                subType=Location.ANY
            )
            if response.data:  # Check if response data is not empty
                iata_code = response.data[0]['iataCode']
                iata_codes[city] = iata_code
            else:
                print(f"Error: Could not find airport code for {city}. No matching locations found.")
        except ResponseError as error:
            print(f"Error: An error occurred while searching for airport code for {city}: {error}")

    for i in range(len(cities) - 1):
        src = cities[i]
        dest = cities[i + 1]
        flights[src] = []

        try:
            # Check if the source city exists in the iata_codes dictionary
            if src in iata_codes:
                src_iata = iata_codes[src]
            else:
                print(f"Error: Could not find airport code for source city {src}.")
                continue

            # Check if the destination city exists in the iata_codes dictionary
            if dest in iata_codes:
                dest_iata = iata_codes[dest]
            else:
                print(f"Error: Could not find airport code for destination city {dest}.")
                continue

            response = amadeus.shopping.flight_offers_search.get(
                originLocationCode=src_iata,
                destinationLocationCode=dest_iata,
                departureDate=dates[i],
                adults=num_adults,
                travelClass='ECONOMY',
                currencyCode='INR',
                max=20
            )

            flights[src].append(response.result['data'])

        except ResponseError as error:
            print(f"Error: List of flights for {src} to {dest} is not available.")

    return flights


def display_flight_info(flight_data):
    city_flight_info = {}  # Dictionary to store flight info by city

    for src, flights in flight_data.items():
        city_flight_info[src] = []  # Initialize list for each city
        for flight_list in flights:
            for flight in flight_list:
                # Extract specific information from the flight data
                departure_time = flight['itineraries'][0]['segments'][0]['departure']['at']
                arrival_time = flight['itineraries'][0]['segments'][-1]['arrival']['at']
                airline = flight['validatingAirlineCodes'][0]
                price = flight['price']['grandTotal']
                currency = flight['price']['currency']

                # Format flight information as a dictionary
                flight_info = {
                    "Airline": airline,
                    "Departure Time": departure_time,
                    "Arrival Time": arrival_time,
                    "Price": price,
                    "Currency": currency
                }
                # Append flight information to the list for the corresponding city
                city_flight_info[src].append(flight_info)

    return city_flight_info


@st.cache_data(show_spinner=False)
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

    print(cities)
    print(dates)

    input_dict['cities'] = cities
    input_dict['dates'] = dates
    all_city_dict = {}

    printables = {}
    city_string = ''
    for city in cities:
        city_string += city + ' - '
    st.subheader("Cities: ")
    st.write(city_string)
    printables['city_string'] = city_string

    for i in range(len(cities)):
        # st.write(cities[i], dates[i], dates[i+1], input_dict['num_adults'], input_dict['num_children'])
        all_city_dict.update(
            get_hotel_data(cities[i], dates[i], dates[i + 1], input_dict['num_adults'], input_dict['num_children']))
    input_dict['hotels_by_city'] = all_city_dict

    # Part 2: Actually generate the itinerary
    user_message = f"Design a detailed itinerary for a trip from {input_dict['src']} to {input_dict['dest']} starting from {input_dict['start_date']} and for " \
                   f"{input_dict['num_days']} days. The ordered list of cities is {cities} and of dates is {dates}. The budget for this trip is {input_dict['price_per_person']} INR per person. This trip is designed " \
                   f"for {input_dict['num_tourists']} mainly with their {input_dict['type_of_travelers']} with an average age of {input_dict['average_age']}.The " \
                   f"primary interests for activities are {input_dict['genre']}.The preferred mode(s) of travel include " \
                   f"{input_dict['mode_of_travel']}.The group prefers {input_dict['food']} food. Please structure the itinerary with a detailed " \
                   f"plan for each day, including activities, locations, weather according to the season they are " \
                   f"travelling and estimated travel distances and times. Write the travel time and distance in the day's subheading. " \
                   f"Ensure to consider the preferences and " \
                   f"interests of the group for each day's schedule.Also consider this note {special_note}. Important considerations: Factor in travel time " \
                   f"between destinations. Suggest local transportation options. Include a mix of activities that cater" \
                   f" to the group's interests. Also add distance of travel for each day and approx time " \
                   f"of travel. Also you can give a name for each day in the itinerary which will be more " \
                   f"appealing. Keep the response descriptive and . Give a title to the itinerary. Do not suggest any activities " \
                   f"in the first city if the travel time and distance is more otherwise we can suggest activities."

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

    st.subheader("Itinerary")
    response = st.write_stream(chat_completion)

    flight_data = flight_search(input_dict)

    # Display flight information
    flight_info = display_flight_info(flight_data)
    content = response

    # Split content into individual days
    days = content.split("\n\n")

    st.session_state['input_dict'] = input_dict
    return response, all_city_dict, flight_info, days, city_string


def fetch_images_when_selected(selected_proper_noun):
    if selected_proper_noun:
        fetch_images_from_pexels([selected_proper_noun])


def fetch_images_from_pexels(proper_noun):
    # Pexels API configuration
    API_KEY = 'HtfomN1StvNLr9SgjXbdC8qE8nuIHzbMXfwmWcHwRe24eNziS6kr5ifC'
    base_url = 'https://api.pexels.com/v1/search'

    # Set up parameters for the request
    params = {'query': proper_noun, 'per_page': 1}

    # Make the request to Pexels API
    response = requests.get(base_url, params=params, headers={'Authorization': API_KEY})
    print("Fetching images for:", proper_noun)

    if response.status_code == 200:
        # Extract image URL from the response
        data = response.json()
        if data['total_results'] > 0:
            image_url = data['photos'][0]['src']['large']
            # Save the image to a file
            save_image(image_url, proper_noun)
        else:
            print(f"No images found for {proper_noun}")
    else:
        print(f"Error fetching images for {proper_noun}")


def save_image(image_url, proper_nouns):
    # Create images directory if not exists
    if not os.path.exists('images'):
        os.makedirs('images')

    # Download and save the image
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(f'images/{proper_nouns}.jpg', 'wb') as f:
            f.write(response.content)
            print(f"Image saved for {proper_nouns}")
    else:
        print(f"Failed to download image for {proper_nouns}")


# Download required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


def extract_proper_nouns(text):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Perform part-of-speech tagging
    tagged_words = pos_tag(words)

    # Perform named entity recognition (NER)
    ne_tree = ne_chunk(tagged_words)

    # Extract proper nouns from NER results
    proper_nouns = []
    for subtree in ne_tree:
        if isinstance(subtree, nltk.Tree) and subtree.label() == 'PERSON':
            # If it's a PERSON entity, treat it as a proper noun
            proper_nouns.append(" ".join([word for word, _ in subtree.leaves()]))
        elif isinstance(subtree, tuple) and subtree[1] == 'NNP':
            # If it's tagged as a proper noun (NNP), add it to the list
            proper_nouns.append(subtree[0])

    return proper_nouns


@st.cache_data(show_spinner=False)
def text_to_doc(itinerary, input_dict):
    document = Document()
    paragraph = document.add_paragraph()
    run = paragraph.add_run()
    run.add_picture("icons/logo.png", width=Inches(2.7))  # Adjust width as needed
    paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

    first_line = itinerary.split('\n')[1]

    # Add the first line as a centered header
    header = document.add_heading(level=1)
    header_run = header.add_run(first_line)
    header_run.font.size = Pt(16)
    header.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

    # Add subheader with small images
    subheader_text = f"{input_dict['num_days']} | {input_dict['start_date']} | {input_dict['dest']} to {input_dict['src']}"
    subheader_paragraph = document.add_paragraph()
    subheader_paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

    # Function to add image and text with specified width
    def add_image_run(paragraph, image_path, text, image_width):
        run = paragraph.add_run()
        run.add_picture(image_path, width=image_width)
        run.add_text(text)

    # Define a list of different image paths
    image_paths = ["icons/cal.png", "icons/global.jpeg", "icons/gps.png"]  # Add more paths as needed

    # Add subheader components with images
    for index, component in enumerate(subheader_text.split(" | ")):
        image_path = image_paths[index] if index < len(image_paths) else image_paths[
            -1]  # Use the last image if there are more components than images
        add_image_run(subheader_paragraph, image_path, component, Inches(0.15))

    paragraph3 = document.add_paragraph()
    run3 = paragraph3.add_run("Main Focus ")
    run3.bold = True
    run3 = paragraph3.add_run(input_dict['genre'])

    paragraph4 = document.add_paragraph()
    run4 = paragraph4.add_run("Commences on: ")
    run4.bold = True
    run4 = paragraph4.add_run(str(input_dict['start_date']))

    paragraph6 = document.add_paragraph()
    run6 = paragraph6.add_run("Budget : ")
    run6.bold = True
    run6 = paragraph6.add_run(str(input_dict['price_per_person']))

    # Add countries and cities covered
    paragraph5 = document.add_paragraph()
    run5 = paragraph5.add_run("Countries and Cities Covered:")
    run5.bold = True

    # Join the city names with a comma
    city_names = ", ".join(input_dict['cities'])

    # Add all city names in a single paragraph
    document.add_paragraph(f"- {city_names}")

    line_paragraph = document.add_paragraph()
    line_run = line_paragraph.add_run(
        "_")

    # Set the font size of the line
    line_font = line_run.font
    line_font.size = Pt(12)

    # Set the paragraph alignment to center
    line_paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

    # Function to check if a specific word is present in the itinerary
    paragraph = document.add_paragraph("")

    paragraph.bold = True

    def contains_word(itinerary, word):
        return word.lower() in itinerary.lower()

    # Function to add image without text
    def add_image_without_text(paragraph, image_path, image_width):
        run = paragraph.add_run()
        run.add_picture(image_path, width=image_width)

    # Check if the word "hotel" is present in the itinerary
    if contains_word(itinerary, "hotel"):
        add_image_without_text(paragraph, "icons/hotel.png", Inches(1))
    if contains_word(itinerary, "flight"):
        add_image_without_text(paragraph, "icons/flight.jpeg", Inches(1))
    if contains_word(itinerary, "eat"):
        add_image_without_text(paragraph, "icons/meal.png", Inches(1))
    if contains_word(itinerary, "attraction"):
        add_image_without_text(paragraph, "icons/sightseeing.png", Inches(1))
    if contains_word(itinerary, "train"):
        add_image_without_text(paragraph, "icons/train.jpeg", Inches(1))
    if contains_word(itinerary, "religious"):
        add_image_without_text(paragraph, "icons/religious.png", Inches(1))
    if contains_word(itinerary, "work"):
        add_image_without_text(paragraph, "icons/work.png", Inches(1))

    paragraph.add_run().add_break()  # Add a blank line

    paragraph.add_run().add_break()  # Add a blank line

    paragraph5 = document.add_paragraph()
    a = paragraph5.add_run("Tour Itinerary :")
    a.bold = True
    for line in itinerary.split('\n')[2:]:
        paragraph = document.add_paragraph()
        image_added = False  # Flag variable to track whether an image has been added to the line
        for char in line:
            if "distance" in line and "Day" not in line and not image_added:
                add_image_without_text(paragraph, "icons/distance.png", Inches(0.4))
                image_added = True
            elif "Dinner" in line and "Day" not in line and not image_added:
                add_image_without_text(paragraph, "icons/meal.png", Inches(0.4))
                image_added = True
            elif "Lunch" in line and "Day" not in line and not image_added:
                add_image_without_text(paragraph, "icons/meal.png", Inches(0.4))
                image_added = True
            elif "Breakfast" in line and "Day" not in line and not image_added:
                add_image_without_text(paragraph, "icons/breakfast.jpeg", Inches(0.5))
                image_added = True
            elif "hotel" in line and "Day" not in line and not image_added:
                add_image_without_text(paragraph, "icons/hotel1.png", Inches(0.4))
                image_added = True
            elif "visit" in line and "Day" not in line and not image_added:
                add_image_without_text(paragraph, "icons/sightseeing.png", Inches(0.3))
                image_added = True
            elif "Visit" in line and "Day" not in line and not image_added:
                add_image_without_text(paragraph, "icons/sightseeing.png", Inches(0.3))
                image_added = True
            elif "arrive" in line and "Day" not in line and not image_added:
                add_image_without_text(paragraph, "icons/work.png", Inches(0.4))
                image_added = True
            elif "Arrive" in line and "Day" not in line and not image_added:
                add_image_without_text(paragraph, "icons/work.png", Inches(0.4))
                image_added = True
            elif "trek" in line and "Day" not in line and not image_added:
                add_image_without_text(paragraph, "icons/religious.png", Inches(0.3))
                image_added = True
            elif "Trek" in line and "Day" not in line and not image_added:
                add_image_without_text(paragraph, "icons/religious.png", Inches(0.3))
                image_added = True
            elif "Distance" in line and "Day" not in line and not image_added:
                add_image_without_text(paragraph, "icons/distance.png", Inches(0.3))
                image_added = True
            elif "Travel" in line and "Day" not in line and not image_added:
                add_image_without_text(paragraph, "icons/distance.png", Inches(0.3))
                image_added = True
            elif "Discover" in line and "Day" not in line and not image_added:
                add_image_without_text(paragraph, "icons/sightseeing.png", Inches(0.3))
                image_added = True
            elif "Explore" in line and "Day" not in line and not image_added:
                add_image_without_text(paragraph, "icons/sightseeing.png", Inches(0.3))
                image_added = True
            elif "sunset" in line and "Day" not in line and not image_added:
                add_image_without_text(paragraph, "icons/sunset.jpeg", Inches(0.3))
                image_added = True
            elif "Taste" in line and "Day" not in line and not image_added:
                add_image_without_text(paragraph, "icons/breakfast.jpeg", Inches(0.5))
                image_added = True
            elif "views" in line and "Day" not in line and not image_added:
                add_image_without_text(paragraph, "icons/sightseeing.png", Inches(0.3))
                image_added = True
            elif "Depart" in line and "Day" not in line and not image_added:
                add_image_without_text(paragraph, "icons/work.png", Inches(0.3))
                image_added = True
            elif "Fly" in line and "Day" not in line and not image_added:
                add_image_without_text(paragraph, "icons/flight.jpeg", Inches(0.3))
                image_added = True
            elif "local" in line and "Day" not in line and not image_added:
                add_image_without_text(paragraph, "icons/local.jpeg", Inches(0.3))
                image_added = True
        for char in line:
            if "Day" in line:
                run = paragraph.add_run(char)
                run.bold = True
                run.italic = True
                run.font.size = Pt(12)
                run.font.name = 'Arial'
                for run in paragraph.runs:
                    run.font.underline = True

            else:
                run = paragraph.add_run(char)

    # Add a table
    table_data = [
        ["Destination", " Hotel"],
    ]
    for city in input_dict['cities']:
        table_data.append([city, ''])

    table = document.add_table(rows=len(table_data), cols=2)

    # adding data to table
    for i, row_data in enumerate(table_data):
        for j, cell_data in enumerate(row_data):
            table.cell(i, j).text = cell_data

    # Apply alignment to the table
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

    # Define a custom table style (optional)
    table.style = 'Table Grid'


    doc_io = BytesIO()
    document.save(doc_io)
    doc_io.seek(0)
    return doc_io


@st.cache_data(show_spinner=False)
def display_image_choices(days):
    proper_nouns_by_day = {}
    for day in days:
        lines = day.split("\n")
        day_name = lines[0]
        day_content = "\n".join(lines[1:])  # Exclude the day name

        # Extract keywords for the day
        keywords = extract_proper_nouns(day_content)
        unique_proper_nouns = list(set(keywords))

        # Store keywords for the day
        proper_nouns_by_day[day_name] = unique_proper_nouns
    st.session_state['proper_nouns_by_day'] = proper_nouns_by_day
    return proper_nouns_by_day

def create_zip_file():
    # Get the list of image files in the images directory
    image_files = [filename for filename in os.listdir("images") if filename.endswith(".jpg")]

    # Create a zip file
    with zipfile.ZipFile("images.zip", "w") as zipf:
        for image_file in image_files:
            zipf.write(os.path.join("images", image_file), image_file)

if st.session_state.get('input_dict', False):
    for key in input_dict.keys():
        if input_dict[key] != st.session_state['input_dict'][key]:
            st.session_state['data_changed'] = True
            break

if st.button("Generate Itinerary", type="primary"):
    null_flag = False
    for key in input_dict.keys():
        if input_dict[key] is None:
            st.warning(f'Please enter {key}!')
            null_flag = True
            break

    if not null_flag:
        generated_itinerary, city_dict, flight_info, days, city_string = generate_itinerary(input_dict)
        st.session_state["cached_data_generated"] = True
        st.session_state['data_changed'] = False
        isGenerated = True

elif st.session_state.get("cached_data_generated", False) and not st.session_state['data_changed']:
    generated_itinerary, city_dict, flight_info, days, city_string = generate_itinerary(input_dict)

if st.session_state.get("cached_data_generated", False) and not st.session_state['data_changed']:
    st.subheader("Hotels")
    for city, hotels in city_dict.items():
        city_expander = st.expander(f"{city}")
        with city_expander:
            for hotel in hotels:
                st.write(f"- {hotel['hotel_name']}")
                st.write(f"  Address: {hotel['address']}")
                st.write(f"  Price per day: {hotel['price']} INR")
                st.write(f"  Rating: {hotel['rating']}")
                # Add more details as needed (amenities, images, etc.)
                st.write("---")  # Separator between hotels\

    st.subheader("Flight Details")
    for city, flights in flight_info.items():
        city_expander = st.expander(f"{city}")
        with city_expander:
            for flight in flights:
                st.write(f"- {flight['Airline']}")
                st.write(f"  Departure Time: {flight['Departure Time']}")
                st.write(f"  Arrival Time: {flight['Arrival Time']}")
                st.write(f"  Price: {flight['Price']} INR")
                # Add more details as needed (amenities, images, etc.)
                st.write("---")

    proper_nouns_by_day = display_image_choices(days)
    for day, proper_nouns in proper_nouns_by_day.items():

        if proper_nouns:
            selected_proper_nouns = None
            st.subheader(f"Choose image for {day}:")
            selected_proper_nouns = st.selectbox(f"Choose image for {day}:", proper_nouns)

            fetch_images_when_selected(selected_proper_nouns)

    doc_io = text_to_doc(generated_itinerary, st.session_state['input_dict'])

    st.download_button(
        label="Download Word Document",
        data=doc_io,
        file_name=f"{input_dict['dest']} Itinerary.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    if st.button("Convert Images as Zip"):
        create_zip_file()
    # Download the zip file
    with open("images.zip", "rb") as f:
        zip_data = f.read()
        st.download_button(
            label="Download Zip",
            data=zip_data,
            file_name="images.zip",
            mime="application/zip"
        )
