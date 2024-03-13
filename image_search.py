#Approach1

# from keybert import KeyBERT

# kw_model = KeyBERT()
# keywords = kw_model.extract_keywords("Visit the Umaid Bhawan Palace and the Mandore Gardens",keyphrase_ngram_range=(1, 1))
# only_keywords = []
# if len(keywords)>3:
#     keywords = keywords[0:3]
# for key in keywords:
#     only_keywords.append(key[0])
# print(only_keywords)

# So start by passing the itinerary as a whole to the gpt 3.5 turbo api. Then it will return the keywords.
# Which will then undergo image search. User selects the keyword which he wants to search for.
# If he is not satisfied with any of the keyword, he has the option of choosing his own keyword to image search.
# The given keyword undergoes google image search and it will return five images in image directory.

import openai
import re
import json
def extract_proper_nouns(itinerary, api_key):
    # Initialize the OpenAI API client
    openai.api_key = api_key

    # Define the prompt
    prompt = "Extract a list of keywords which are Proper Nouns from the following itinerary:\n" + itinerary

    # Call the GPT-3.5 turbo API
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": prompt},
      ],
    )

    proper_nouns = re.findall(r'\b[A-Z][a-z]*\s[A-Z][a-z]*\b|\b[A-Z][a-z]*\b', response['choices'][0]['message']['content'])


    # Extract the Proper Nouns from the response
    proper_nouns_dict = {f"keyword_{i+1}": word for i, word in enumerate(proper_nouns)}

    # Convert the dictionary to JSON format

    with open(output_file, 'w') as f:
        json.dump(proper_nouns_dict, f, indent=4)

    # proper_nouns_json = json.dumps(proper_nouns_dict, indent=4)
    # return proper_nouns_json

# Set your API key
api_key = ""

# Define the itinerary
itinerary = '''
Day 2: Jaipur Exploration - Discovering the Royal Heritage
- Visit the Amber Fort and explore its stunning architecture
- Head to the City Palace and Jantar Mantar for more sightseeing
- Enjoy a delicious Pure Veg lunch at a local restaurant
- Visit the Hawa Mahal for some great photo opportunities
- Spend the evening shopping at the local markets
- Dine at a cozy restaurant and retire for the night

Day 3: Jaipur to Jodhpur - Journey to the Blue City
- Distance: Approximately 340 km
- Mode of Transport: Bus
- Check into a budget-friendly hotel or guesthouse in Jodhpur
- Explore the Mehrangarh Fort and Jaswant Thada
- Enjoy a Pure Veg dinner at a local restaurant
- Rest and prepare for the next day's adventures
'''
output_file = "keywords.json"

# Extract Proper Nouns from the itinerary
proper_nouns_json = extract_proper_nouns(itinerary, api_key)

from google_images_search import GoogleImagesSearch
import json
import os
# Load the keywords from the JSON file
with open('keywords.json', 'r') as f:
    keywords = json.load(f)

# Display the keywords and let the user choose
print("Select a keyword to search for images:")
for i, keyword in enumerate(keywords, 1):
    print(f"{i}. {keyword}")

# Allow the user to select a keyword
selected_index = int(input("Enter the number of the keyword you want to search for, or enter 0 to type your own keyword: "))
if selected_index == 0:
    selected_keyword = input("Enter the keyword you want to search for: ")
else:
    selected_keyword = keywords[selected_index - 1]

save_dir = 'images/' + selected_keyword.replace(' ', '_')
os.makedirs(save_dir, exist_ok=True)
gis = GoogleImagesSearch('API', 'ORG-ID')
_search_params = {
    'q': selected_keyword,
    'num': 1,
    'fileType': 'jpg|gif|png',
    'rights': 'cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived'
}

# Perform the image search
gis.search(search_params=_search_params, path_to_dir=save_dir)