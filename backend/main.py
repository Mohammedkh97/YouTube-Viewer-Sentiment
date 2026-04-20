import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.dates as mdates
import pickle

app = FastAPI(title="YouTube Viewer Sentiment API")

# Enable CORS for all routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CommentWithTimestamp(BaseModel):
    text: str
    timestamp: str

class PredictWithTimestampsRequest(BaseModel):
    comments: List[CommentWithTimestamp]

class PredictRequest(BaseModel):
    comments: List[str]

class SentimentCountsRequest(BaseModel):
    sentiment_counts: Dict[str, int]

class SentimentData(BaseModel):
    sentiment: str
    timestamp: str
    comment: Optional[str] = None

class TrendGraphRequest(BaseModel):
    sentiment_data: List[SentimentData]


# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment


def load_model(model_path, vectorizer_path):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
      
        return model, vectorizer
    except Exception as e:
        raise


import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Initialize the model and vectorizer
model, vectorizer = load_model(os.path.join(BASE_DIR, "lgbm_model.pkl"), os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))

@app.get('/')
def home():
    return "Welcome to our FastAPI"


@app.post('/predict_with_timestamps')
def predict_with_timestamps(request: PredictWithTimestampsRequest):
    comments_data = request.comments
    
    if not comments_data:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        comments = [item.text for item in comments_data]
        timestamps = [item.timestamp for item in comments_data]

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        dense_comments = transformed_comments.toarray()
        
        predictions = model.predict(dense_comments).tolist()
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    response = [{"comment": c, "sentiment": s, "timestamp": t} for c, s, t in zip(comments, predictions, timestamps)]
    return response


@app.post('/predict')
def predict(request: PredictRequest):
    comments = request.comments
    print("i am the comment: ", comments)
    print("i am the comment type: ", type(comments))
    
    if not comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        dense_comments = transformed_comments.toarray()
        predictions = model.predict(dense_comments).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    response = [{"comment": c, "sentiment": s} for c, s in zip(comments, predictions)]
    return response


@app.post('/generate_chart')
def generate_chart(request: SentimentCountsRequest):
    try:
        sentiment_counts = request.sentiment_counts
        
        if not sentiment_counts:
            raise HTTPException(status_code=400, detail="No sentiment counts provided")

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal') 

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {str(e)}")


@app.post('/generate_wordcloud')
def generate_wordcloud(request: PredictRequest):
    try:
        comments = request.comments

        if not comments:
            raise HTTPException(status_code=400, detail="No comments provided")

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        text = ' '.join(preprocessed_comments)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word cloud generation failed: {str(e)}")


@app.post('/generate_trend_graph')
def generate_trend_graph(request: TrendGraphRequest):
    try:
        sentiment_data = request.sentiment_data

        if not sentiment_data:
            raise HTTPException(status_code=400, detail="No sentiment data provided")

        df = pd.DataFrame([s.dict() for s in sentiment_data])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)

        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        monthly_counts = df.resample('ME')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        
        # Calculate percentages, handle division by zero
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100
        monthly_percentages.fillna(0, inplace=True)

        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))
        colors = {-1: 'red', 0: 'gray', 1: 'green'}

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend graph generation failed: {str(e)}")

# Remove app.run so we use uvicorn to run it instead
