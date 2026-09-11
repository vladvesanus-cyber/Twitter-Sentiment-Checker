import re

def clean_text(text):
    text = re.sub(r'@\w+', '', text) #Remove mentions
    text = re.sub(r'http\S+|www\.\S+','', text) #Remove URLs
    text = re.sub(r'#\w+', '', text) #Remove hashtags
    text = re.sub(r'\s+', ' ', text).strip() #Remove extra whitespace
    text = text.lower()
    return text

def clean_data(df):
    df['text'] = df['text'].apply(clean_text)
    return df