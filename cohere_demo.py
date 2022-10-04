import cohere
import time
import pandas as pd
import streamlit as st

from cohere.classify import Example

st.set_page_config(page_title="Company describer", page_icon="📰")

API_KEY = 'LBJMAOp2teesYwwmwKMDNXdCo2CEvW1qWLbhiWk9'  # API key from the Cohere dashboard
co = cohere.Client(API_KEY)


input_company = st.text_input("Company name to describe")

if input_company:
  prompt = 'Company: Google\n\nDescription: Google is an American multinational technology company that focuses on search engine technology, online advertising, cloud computing, computer software, quantum computing, e-commerce, artificial intelligence,[9] and consumer electronics.\n--\nCompany: Home Depot\n\nDescription: The Home Depot, Inc., is an American multinational home improvement retail corporation that sells tools, construction products, appliances, and services.\n--\nCompany: HireEz\n\nDescription: hireEZ provides the solution for outbound recruiting with our AI-powered platform. \n--\nCompany: Roblox\n\nDescription: Roblox is an online game platform and game creation system developed by Roblox Corporation that allows users to program games and play games created by other users.\n--\nCompany: ' + input_company + '\n\nDescription:'

  n_generations = 5

  prediction = co.generate( 
    model='large', 
    prompt=prompt, 
    max_tokens=30, 
    temperature=0.6, 
    k=0, 
    p=1, 
    frequency_penalty=0, 
    presence_penalty=0, 
    stop_sequences=["--"], 
    return_likelihoods='NONE') 
  st.header(prediction.generations[0].text)

  _ = st.button("Another company description 📰")