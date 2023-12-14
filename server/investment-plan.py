from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
import json
import requests
import os

from dotenv import load_dotenv

load_dotenv()


def coins_list(_dict):
    api_key = os.getenv('CG_KEY')

    url = ('https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=5&page=1'
           '&sparkline=false&locale=en&x_cg_demo_api_key=' + api_key)
    coins = requests.get(url)

    if coins.status_code == 200:
        print("Got the top 5 from Coin Gecko!")
        data = coins.json()

        # trim out the name, so we don't send unnecessary tokens to the llm
        for item in data:
            if "name" in item:
                del item["name"]

        modified_json_string = json.dumps(data, indent=4)
        return modified_json_string
    else:
        print(f"Error: {coins.status_code}")


# def execute_order(_dict):
#     json_string = _dict.content
#     return json_string


picker_prompt = ChatPromptTemplate.from_template("Here is a list of the top 5 crypto currencies by market cap from "
                                                 "the coin gecko api '''{coins}'''."
                                                 "Given your knowledge of investing create an investment plan for an "
                                                 "individual that ONLY has $3,000 using these currencies."
                                                 "Invest as much of the amount without going over. Do not allocate more"
                                                 "than the $3,000!"
                                                 " In the response return the price, ticker, id and the percentage "
                                                 "and dollar allocation that should be allocated to each currency. "
                                                 "ONLY return JSON, do NOT make up any currencies that aren't in the "
                                                 "list and do not narrate. Ensure the percentages sum to 100%. "
                                                 "ALWAYS use this json structure, create a top level attribute called"
                                                 "investment_plan that has an array of objects that contain the "
                                                 "following fields id, price, ticker, percentage_allocation, and "
                                                 "dollar_allocation.  Always use those attribute names and return "
                                                 "valid JSON!"
                                                 )
model = ChatOpenAI()

chain = (
        {
            "coins": RunnableLambda(coins_list)
        }
        | picker_prompt
        | model
        # | RunnableLambda(execute_order)
)

response = chain.invoke({})

# Convert response.content from JSON string to dictionary
try:
    response_data = json.loads(response.content)
except json.JSONDecodeError:
    print("Error decoding JSON from response.content")
    response_data = {}

# Check if response_data has the expected structure
if 'investment_plan' in response_data:
    total_dollar_allocation = 0
    coin_quantities = {}

    for item in response_data['investment_plan']:
        if all(key in item for key in ('dollar_allocation', 'price', 'ticker', 'percentage_allocation')):
            print(f"Investing in {item['ticker']} the current price is ${item['price']}; "
                  f"percentage of total investment is {item['percentage_allocation']}%;"
                  f" the dollar allocation will be ${item['dollar_allocation']}.")
            total_dollar_allocation += item['dollar_allocation']
            coin_quantity = item['dollar_allocation'] / item['price']
            coin_quantities[item['ticker']] = coin_quantity
    print("")
    print(f"Total Investment Amount: {total_dollar_allocation}")
    print("")
    print("Quantities of Each Coin to be Purchased:")
    for ticker, quantity in coin_quantities.items():
        print(f"{ticker}: amount = {quantity}")
else:
    print("The response does not contain an 'investment_plan'")
