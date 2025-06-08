# Reddit Guard

A tool to analyze the toxicity levels of Reddit communities and help users make informed decisions about joining them.

## Features

- Fetch and analyze posts and comments from any Reddit subreddit
- Calculate toxicity scores using advanced NLP models
- Generate visualizations of toxicity trends
- Provide detailed insights about community behavior
- User-friendly web interface

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Reddit API credentials:
   ```
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   REDDIT_USER_AGENT=your_user_agent
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Enter a subreddit name in the interface
3. View toxicity analysis and insights

## Project Structure

- `app.py`: Main Streamlit application
- `reddit_scraper.py`: Reddit data collection module
- `toxicity_analyzer.py`: NLP-based toxicity analysis
- `utils.py`: Helper functions
- `models/`: Pre-trained models and model-related code
