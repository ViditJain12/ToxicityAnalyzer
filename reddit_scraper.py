import os
import praw
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

class RedditScraper:
    def __init__(self):
        load_dotenv()
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )

    def get_subreddit_data(self, subreddit_name, time_filter='month', limit=100):
        """
        Fetch posts and comments from a subreddit
        
        Args:
            subreddit_name (str): Name of the subreddit
            time_filter (str): One of (hour, day, week, month, year, all)
            limit (int): Maximum number of posts to fetch
            
        Returns:
            tuple: (posts_df, comments_df)
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Get posts
            posts = []
            for post in subreddit.top(time_filter=time_filter, limit=limit):
                posts.append({
                    'id': post.id,
                    'text': post.title + ' ' + post.selftext,
                    'score': post.score,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'author': str(post.author),
                    'type': 'post',
                    'title': post.title,
                    'upvote_ratio': post.upvote_ratio,
                    'num_comments': post.num_comments,
                    'url': post.url
                })
            
            # Get comments
            comments = []
            for post in subreddit.top(time_filter=time_filter, limit=limit):
                post.comments.replace_more(limit=0)  # Get only top-level comments
                for comment in post.comments:
                    comments.append({
                        'id': comment.id,
                        'text': comment.body,
                        'score': comment.score,
                        'created_utc': datetime.fromtimestamp(comment.created_utc),
                        'author': str(comment.author),
                        'type': 'comment',
                        'post_id': post.id
                    })
            
            # Create DataFrames
            posts_df = pd.DataFrame(posts)
            comments_df = pd.DataFrame(comments)
            
            return posts_df, comments_df
            
        except Exception as e:
            print(f"Error fetching data from r/{subreddit_name}: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

    def get_subreddit_stats(self, subreddit_name):
        """
        Get basic statistics about a subreddit
        
        Args:
            subreddit_name (str): Name of the subreddit
            
        Returns:
            dict: Subreddit statistics
        """
        subreddit = self.reddit.subreddit(subreddit_name)
        
        return {
            'name': subreddit_name,
            'subscribers': subreddit.subscribers,
            'active_users': subreddit.active_user_count,
            'description': subreddit.description,
            'created_utc': datetime.fromtimestamp(subreddit.created_utc),
            'over18': subreddit.over18
        } 