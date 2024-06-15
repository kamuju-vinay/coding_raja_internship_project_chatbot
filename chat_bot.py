import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

# Sample dataset of conversations
conversations = {
    "greetings": ["hello", "hi", "hey", "good morning", "good evening", "howdy", "hiya", "what's up", "greetings", "good afternoon"],
    "goodbyes": ["goodbye", "bye", "see you later", "take care", "farewell", "catch you later", "later", "see ya", "adios", "bye bye"],
    "thanks": ["thank you", "thanks", "i appreciate it", "much obliged", "thanks a lot", "thanks so much", "thanks a bunch", "cheers", "thankful", "grateful"],
    "questions": ["how are you", "how's it going", "what's up", "how have you been", "how's life", "how are things", "what's new", "how's everything", "what's going on", "how's your day"],
    "compliments": ["you are great", "you are awesome", "you are amazing", "you are the best", "great job", "well done", "fantastic", "you are incredible", "you are wonderful", "you rock"],
    "apologies": ["i'm sorry", "sorry", "my apologies", "forgive me", "i apologize", "pardon me", "i'm sorry for that", "sorry about that", "please forgive me", "i regret that"],
    "information": ["tell me more", "i need information", "give me details", "i need to know", "can you explain", "can you tell me", "i need help", "give me more info", "help me understand", "provide details"],
    "weather": ["what's the weather like", "how's the weather", "is it going to rain", "what's the forecast", "tell me the weather", "weather update", "is it sunny", "weather report", "weather conditions", "current weather"],
    "time": ["what time is it", "current time", "can you tell me the time", "time please", "what's the time", "do you have the time", "tell me the time", "time now", "current time update", "time of day"],
    "date": ["what's the date", "current date", "can you tell me the date", "date please", "what's today's date", "today's date", "date now", "current date update", "day and date", "date information"],
    "help": ["i need help", "can you help me", "assist me", "i need assistance", "help me out", "i need support", "can you assist", "help required", "need help", "support me"],
    "jokes": ["tell me a joke", "make me laugh", "do you know any jokes", "tell a funny joke", "joke please", "can you tell a joke", "funny joke", "give me a joke", "i want to hear a joke", "joke time"],
    "news": ["tell me the news", "what's in the news", "current news", "news update", "latest news", "news report", "any news", "news today", "breaking news", "news headlines"],
    "sports": ["sports update", "latest sports news", "who won the game", "sports scores", "sports news", "tell me about sports", "sports report", "game results", "sports highlights", "sports information"],
    "responses": {
        "greetings": "Hello! How can I assist you today?",
        "goodbyes": "Goodbye! Have a great day!",
        "thanks": "You're welcome! How can I help you further?",
        "questions": "I'm doing great, thank you! How can I assist you today?",
        "compliments": "Thank you! I'm here to help you with anything you need.",
        "apologies": "No problem at all! How can I assist you further?",
        "information": "Sure, I'd be happy to help. What do you need information about?",
        "weather": "The current weather is sunny with a slight breeze. How can I help you further?",
        "time": "It's 3:45 PM right now. What else can I assist you with?",
        "date": "Today is June 15th, 2024. How can I help you today?",
        "help": "Of course! What do you need help with?",
        "jokes": "Why don't scientists trust atoms? Because they make up everything!",
        "news": "Today's top news: AI technology is advancing rapidly, changing various industries.",
        "sports": "In sports news, the local team won their game last night with a score of 3-2."
    }
}

# Function to preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

# Function to recognize intent
def recognize_intent(text, conversations):
    processed_text = preprocess_text(text)
    #print(f"Processed text: {processed_text}")  # Debugging statement
    
    for intent, phrases in conversations.items():
        if intent == "responses":
            continue
        for phrase in phrases:
            if set(preprocess_text(phrase)) == set(processed_text):  # Exact match
               # print(f"Recognized exact match intent: {intent}")  # Debugging statement
                return intent

    # Fallback to keyword matching
    for intent, phrases in conversations.items():
        if intent == "responses":
            continue
        for phrase in phrases:
            if set(preprocess_text(phrase)).intersection(set(processed_text)):
                print(f"Recognized keyword match intent: {intent}")  # Debugging statement
                return intent
                
    return None

# Function to generate response
def generate_response(intent, conversations):
    if intent in conversations["responses"]:
        return conversations["responses"][intent]
    return "I'm not sure how to respond to that. Can you please clarify?"

# Chatbot interaction function
def chatbot_response(user_input, conversations):
    intent = recognize_intent(user_input, conversations)
    if intent:
        return generate_response(intent, conversations)
    else:
        return "I'm sorry, I didn't understand that. Could you please rephrase?"

# Example interactions with the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Goodbye! Have a great day!")
        break
    response = chatbot_response(user_input, conversations)
    print(f"Chatbot: {response}")
