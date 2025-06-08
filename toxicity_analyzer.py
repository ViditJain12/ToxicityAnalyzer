import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import re
from datetime import datetime, timedelta

class ToxicityAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
        self.model = AutoModelForSequenceClassification.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Define toxicity thresholds
        self.HIGH_TOXICITY_THRESHOLD = 0.7
        self.MEDIUM_TOXICITY_THRESHOLD = 0.4

    def analyze_text(self, text):
        """
        Analyze a single text for toxicity
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Toxicity score between 0 and 1
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)
            toxicity_score = scores[0][1].item()  # Probability of toxic class
            
        return toxicity_score

    def analyze_dataframe(self, df, text_column):
        """
        Analyze toxicity for all texts in a DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame containing texts
            text_column (str): Name of the column containing texts
            
        Returns:
            pd.DataFrame: Original DataFrame with added toxicity scores
        """
        df['toxicity_score'] = df[text_column].apply(self.analyze_text)
        return df

    def get_toxicity_metrics(self, df):
        """
        Calculate comprehensive toxicity metrics from analyzed data
        
        Args:
            df (pd.DataFrame): DataFrame with toxicity scores
            
        Returns:
            dict: Dictionary of toxicity metrics
        """
        return {
            'mean_toxicity': df['toxicity_score'].mean(),
            'median_toxicity': df['toxicity_score'].median(),
            'std_toxicity': df['toxicity_score'].std(),
            'max_toxicity': df['toxicity_score'].max(),
            'min_toxicity': df['toxicity_score'].min(),
            'toxic_percentage': (df['toxicity_score'] > 0.5).mean() * 100,
            'highly_toxic_percentage': (df['toxicity_score'] > self.HIGH_TOXICITY_THRESHOLD).mean() * 100,
            'medium_toxic_percentage': ((df['toxicity_score'] > self.MEDIUM_TOXICITY_THRESHOLD) & 
                                      (df['toxicity_score'] <= self.HIGH_TOXICITY_THRESHOLD)).mean() * 100,
            'low_toxic_percentage': (df['toxicity_score'] <= self.MEDIUM_TOXICITY_THRESHOLD).mean() * 100
        }

    def get_toxicity_trends(self, df, time_column, freq='D'):
        """
        Calculate toxicity trends over time
        
        Args:
            df (pd.DataFrame): DataFrame with toxicity scores
            time_column (str): Name of the column containing timestamps
            freq (str): Frequency for resampling ('D' for daily, 'W' for weekly)
            
        Returns:
            pd.DataFrame: DataFrame with time-based toxicity metrics
        """
        df[time_column] = pd.to_datetime(df[time_column])
        df.set_index(time_column, inplace=True)
        
        trends = df.resample(freq).agg({
            'toxicity_score': ['mean', 'std', 'count'],
            'text': 'count'
        }).reset_index()
        
        trends.columns = ['date', 'mean_toxicity', 'std_toxicity', 'comment_count', 'text_count']
        return trends

    def get_toxic_word_cloud(self, df, text_column, min_toxicity=0.4):
        """
        Generate word cloud from toxic comments
        
        Args:
            df (pd.DataFrame): DataFrame with texts and toxicity scores
            text_column (str): Name of the column containing texts
            min_toxicity (float): Minimum toxicity score to consider (default: 0.4)
            
        Returns:
            WordCloud: WordCloud object or None if no toxic content found
        """
        # Filter toxic comments
        toxic_texts = df[df['toxicity_score'] >= min_toxicity][text_column]
        
        if len(toxic_texts) == 0:
            return None
        
        # Combine all texts
        text = ' '.join(toxic_texts.astype(str))
        
        # Clean text
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            contour_width=3,
            contour_color='steelblue'
        ).generate(text)
        
        return wordcloud

    def get_comment_volume_toxicity_correlation(self, df, time_column):
        """
        Calculate correlation between comment volume and toxicity
        
        Args:
            df (pd.DataFrame): DataFrame with texts and toxicity scores
            time_column (str): Name of the column containing timestamps
            
        Returns:
            tuple: (correlation coefficient, volume_toxicity_df)
        """
        if time_column not in df.columns:
            print(f"Available columns: {df.columns.tolist()}")
            raise ValueError(f"Column '{time_column}' not found in DataFrame")
            
        # Ensure the time column is datetime
        df[time_column] = pd.to_datetime(df[time_column])
        
        # Group by time periods and calculate metrics
        volume_toxicity = df.groupby(df[time_column].dt.date).agg({
            'toxicity_score': 'mean',
            'text': 'count'
        }).reset_index()
        
        volume_toxicity.columns = ['date', 'mean_toxicity', 'comment_count']
        
        # Calculate correlation
        correlation = volume_toxicity['mean_toxicity'].corr(volume_toxicity['comment_count'])
        
        return correlation, volume_toxicity

    def get_top_toxic_comments(self, df, text_column, n=10):
        """
        Get the most toxic comments
        
        Args:
            df (pd.DataFrame): DataFrame with texts and toxicity scores
            text_column (str): Name of the column containing texts
            n (int): Number of top toxic comments to return
            
        Returns:
            pd.DataFrame: DataFrame with top toxic comments
        """
        return df.nlargest(n, 'toxicity_score')[[text_column, 'toxicity_score']]

    def get_toxicity_distribution(self, df):
        """
        Calculate toxicity score distribution
        
        Args:
            df (pd.DataFrame): DataFrame with toxicity scores
            
        Returns:
            pd.DataFrame: DataFrame with toxicity score distribution
        """
        bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
        labels = [f'{i:.1f}-{i+0.1:.1f}' for i in np.arange(0, 1, 0.1)]
        
        df['toxicity_bin'] = pd.cut(df['toxicity_score'], bins=bins, labels=labels)
        distribution = df['toxicity_bin'].value_counts().sort_index()
        
        return distribution 