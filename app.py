import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from reddit_scraper import RedditScraper
from toxicity_analyzer import ToxicityAnalyzer
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
from datetime import datetime

st.set_page_config(page_title="Reddit Toxicity Analyzer", layout="wide")

st.title("Reddit Toxicity Analyzer")
st.write("Analyze the toxicity levels of Reddit communities to make informed decisions about joining them.")

def get_safety_rating(mean_toxicity, highly_toxic_percentage):
    """Calculate safety rating based on toxicity metrics."""
    # Calculate toxic score (0-100, where 0 is least toxic and 100 is most toxic)
    toxic_score = int((mean_toxicity * 0.7 + (highly_toxic_percentage / 100) * 0.3) * 100)
    
    if mean_toxicity < 0.2 and highly_toxic_percentage < 5:
        return {
            "rating": "Very Safe",
            "emoji": "🟢",
            "color": "green",
            "description": "This community maintains a very positive and respectful environment.",
            "recommendation": "Great community to join!",
            "toxic_score": toxic_score
        }
    elif mean_toxicity < 0.3 and highly_toxic_percentage < 10:
        return {
            "rating": "Safe",
            "emoji": "🟡",
            "color": "yellow",
            "description": "This community is generally positive with occasional heated discussions.",
            "recommendation": "Good community to join, but be prepared for occasional debates.",
            "toxic_score": toxic_score
        }
    elif mean_toxicity < 0.4 and highly_toxic_percentage < 15:
        return {
            "rating": "Moderately Toxic",
            "emoji": "🟠",
            "color": "orange",
            "description": "This community has some toxic elements but is still manageable.",
            "recommendation": "Join with caution and be prepared to encounter some negativity.",
            "toxic_score": toxic_score
        }
    else:
        return {
            "rating": "Highly Toxic",
            "emoji": "🔴",
            "color": "red",
            "description": "This community has significant toxicity issues.",
            "recommendation": "Consider avoiding this community or proceed with extreme caution.",
            "toxic_score": toxic_score
        }

def get_similar_communities(subreddit_name, stats):
    """Get similar communities based on subreddit characteristics."""
    # Dictionary of subreddit categories and their related communities
    subreddit_categories = {
        # Programming & Technology
        'python': ['programming', 'learnprogramming', 'coding', 'computerscience', 'technology'],
        'programming': ['python', 'coding', 'learnprogramming', 'computerscience', 'technology'],
        'coding': ['python', 'programming', 'learnprogramming', 'computerscience', 'technology'],
        'learnprogramming': ['python', 'programming', 'coding', 'computerscience', 'technology'],
        'computerscience': ['python', 'programming', 'coding', 'learnprogramming', 'technology'],
        'technology': ['python', 'programming', 'coding', 'computerscience', 'technews'],
        
        # Mobile Games
        'brawlstars': ['clashroyale', 'clashofclans', 'supercell', 'mobilegaming', 'gaming'],
        'clashroyale': ['brawlstars', 'clashofclans', 'supercell', 'mobilegaming', 'gaming'],
        'clashofclans': ['brawlstars', 'clashroyale', 'supercell', 'mobilegaming', 'gaming'],
        'supercell': ['brawlstars', 'clashroyale', 'clashofclans', 'mobilegaming', 'gaming'],
        'mobilegaming': ['brawlstars', 'clashroyale', 'clashofclans', 'supercell', 'gaming'],
        
        # Popular Games
        'minecraft': ['gaming', 'minecraftbuilds', 'minecraftmemes', 'gaming', 'minecraftsuggestions'],
        'fortnite': ['gaming', 'fortnitebr', 'fortnitemobile', 'gaming', 'fortnitecompetitive'],
        'gaming': ['games', 'pcgaming', 'gamingnews', 'gamingmemes', 'gaming'],
        'games': ['gaming', 'pcgaming', 'gamingnews', 'gamingmemes', 'gaming']
    }
    
    # Default suggestions based on category
    default_suggestions = {
        'programming': ['programming', 'coding', 'learnprogramming'],
        'gaming': ['gaming', 'games', 'pcgaming']
    }
    
    # Try to find the subreddit's category
    subreddit_name_lower = subreddit_name.lower()
    
    # Get similar subreddits based on category
    similar_subreddits = []
    if subreddit_name_lower in subreddit_categories:
        similar_subreddits = subreddit_categories[subreddit_name_lower]
    elif any(tech_term in subreddit_name_lower for tech_term in ['python', 'programming', 'coding', 'tech', 'computer']):
        similar_subreddits = default_suggestions['programming']
    elif any(game_term in subreddit_name_lower for game_term in ['game', 'gaming', 'play', 'brawl', 'clash']):
        similar_subreddits = default_suggestions['gaming']
    else:
        similar_subreddits = default_suggestions['programming']
    
    # Remove the current subreddit from suggestions
    similar_subreddits = [sub for sub in similar_subreddits if sub != subreddit_name_lower]
    
    # Fetch real-time stats for each similar subreddit
    similar = []
    for sub in similar_subreddits[:3]:  # Limit to top 3
        try:
            sub_stats = scraper.get_subreddit_stats(sub)
            similar.append({
                "name": sub,
                "subscribers": sub_stats['subscribers']
            })
        except Exception as e:
            # If we can't get stats for a subreddit, skip it
            continue
    
    return similar

def get_community_insights(posts_df, comments_df, stats):
    """Generate interesting insights about the community."""
    insights = []
    
    # Community engagement
    avg_comments = posts_df['num_comments'].mean()
    insights.append(f"💬 Average of {avg_comments:.1f} comments per post")
    
    # Content quality
    avg_score = posts_df['score'].mean()
    insights.append(f"⭐ Average post score: {avg_score:.1f}")
    
    # Community growth
    if stats['subscribers'] > 1000000:
        insights.append("🚀 Large community with over 1M subscribers")
    elif stats['subscribers'] > 100000:
        insights.append("📈 Growing community with over 100K subscribers")
    
    return insights

# Initialize components
scraper = RedditScraper()
analyzer = ToxicityAnalyzer()

# Sidebar inputs
st.sidebar.header("Analysis Parameters")
analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    ["Single Subreddit", "Compare Subreddits"]
)

time_filter = st.sidebar.selectbox(
    "Time Period",
    ["hour", "day", "week", "month", "year", "all"],
    index=3
)
post_limit = st.sidebar.slider("Number of Posts to Analyze", 10, 100, 50)

if analysis_mode == "Single Subreddit":
    subreddit_name = st.sidebar.text_input("Subreddit Name", "python")
    
    if st.sidebar.button("Analyze Subreddit"):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Get subreddit stats
            status_text.text("Fetching subreddit information...")
            stats = scraper.get_subreddit_stats(subreddit_name)
            progress_bar.progress(20)
            
            # Get posts and comments
            status_text.text("Fetching posts and comments...")
            posts_df, comments_df = scraper.get_subreddit_data(
                subreddit_name,
                time_filter=time_filter,
                limit=post_limit
            )
            progress_bar.progress(40)
            
            if posts_df.empty and comments_df.empty:
                st.error(f"No data found for subreddit r/{subreddit_name}")
                st.stop()
            
            # Analyze toxicity
            status_text.text("Analyzing toxicity in posts...")
            posts_df = analyzer.analyze_dataframe(posts_df, 'text')
            progress_bar.progress(60)
            
            status_text.text("Analyzing toxicity in comments...")
            comments_df = analyzer.analyze_dataframe(comments_df, 'text')
            progress_bar.progress(80)
            
            # Calculate metrics
            status_text.text("Calculating metrics and generating visualizations...")
            posts_metrics = analyzer.get_toxicity_metrics(posts_df)
            comments_metrics = analyzer.get_toxicity_metrics(comments_df)
            
            # Get toxicity distribution
            posts_distribution = analyzer.get_toxicity_distribution(posts_df)
            comments_distribution = analyzer.get_toxicity_distribution(comments_df)

            # Calculate overall safety rating
            overall_mean_toxicity = (posts_metrics['mean_toxicity'] + comments_metrics['mean_toxicity']) / 2
            overall_highly_toxic = (posts_metrics['highly_toxic_percentage'] + comments_metrics['highly_toxic_percentage']) / 2
            safety_rating = get_safety_rating(overall_mean_toxicity, overall_highly_toxic)
            
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            st.success("Analysis completed successfully!")

            # Display safety rating
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"<h1 style='text-align: center;'>{safety_rating['emoji']} {safety_rating['rating']} {safety_rating['emoji']}</h1>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: {safety_rating['color']}; font-size: 20px;'>{safety_rating['description']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; font-weight: bold; font-size: 18px;'>{safety_rating['recommendation']}</p>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='text-align: center;'>Toxic Score: {safety_rating['toxic_score']}/100</h2>", unsafe_allow_html=True)
            st.markdown("---")
            
            # Community Insights
            st.subheader("Community Insights")
            insights = get_community_insights(posts_df, comments_df, stats)
            cols = st.columns(len(insights))
            for col, insight in zip(cols, insights):
                col.info(insight)
            
            # Similar Communities
            st.subheader("Similar Communities")
            similar = get_similar_communities(subreddit_name, stats)
            for community in similar:
                st.write(f"r/{community['name']} - {community['subscribers']:,} subscribers")
            
            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Toxicity Analysis", "Trends", "Content Analysis", "Historical Trends", "Word Cloud Analysis"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Subreddit Information")
                    st.write(f"**Name:** r/{stats['name']}")
                    st.write(f"**Subscribers:** {stats['subscribers']:,}")
                    st.write(f"**Active Users:** {stats['active_users']:,}")
                    st.write(f"**Created:** {stats['created_utc'].strftime('%Y-%m-%d')}")
                    st.write(f"**NSFW:** {'Yes' if stats['over18'] else 'No'}")
                    
                    st.subheader("Posts Analysis")
                    st.write(f"**Mean Toxicity:** {posts_metrics['mean_toxicity']:.2%}")
                    st.write(f"**Highly Toxic Posts:** {posts_metrics['highly_toxic_percentage']:.1f}%")
                    st.write(f"**Medium Toxic Posts:** {posts_metrics['medium_toxic_percentage']:.1f}%")
                    st.write(f"**Low Toxic Posts:** {posts_metrics['low_toxic_percentage']:.1f}%")
                    
                with col2:
                    st.subheader("Comments Analysis")
                    st.write(f"**Mean Toxicity:** {comments_metrics['mean_toxicity']:.2%}")
                    st.write(f"**Highly Toxic Comments:** {comments_metrics['highly_toxic_percentage']:.1f}%")
                    st.write(f"**Medium Toxic Comments:** {comments_metrics['medium_toxic_percentage']:.1f}%")
                    st.write(f"**Low Toxic Comments:** {comments_metrics['low_toxic_percentage']:.1f}%")
                    
                    # Toxicity distribution plot
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=comments_distribution.index,
                        y=comments_distribution.values,
                        name='Comments'
                    ))
                    fig.update_layout(
                        title='Toxicity Score Distribution',
                        xaxis_title='Toxicity Score Range',
                        yaxis_title='Number of Comments',
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Most Toxic Posts")
                    toxic_posts = analyzer.get_top_toxic_comments(posts_df, 'text', n=5)
                    for _, post in toxic_posts.iterrows():
                        with st.expander(f"Toxicity: {post['toxicity_score']:.2%}"):
                            st.write(post['text'])
                    
                with col2:
                    st.subheader("Most Toxic Comments")
                    toxic_comments = analyzer.get_top_toxic_comments(comments_df, 'text', n=5)
                    for _, comment in toxic_comments.iterrows():
                        with st.expander(f"Toxicity: {comment['toxicity_score']:.2%}"):
                            st.write(comment['text'])
            
            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Toxicity score distribution by post type
                    fig = go.Figure()
                    fig.add_trace(go.Box(
                        y=posts_df['toxicity_score'],
                        name='Posts',
                        boxpoints='all'
                    ))
                    fig.add_trace(go.Box(
                        y=comments_df['toxicity_score'],
                        name='Comments',
                        boxpoints='all'
                    ))
                    fig.update_layout(
                        title='Toxicity Score Distribution by Type',
                        yaxis_title='Toxicity Score',
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Toxicity score distribution by score
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=posts_df['score'],
                        y=posts_df['toxicity_score'],
                        mode='markers',
                        name='Posts'
                    ))
                    fig.add_trace(go.Scatter(
                        x=comments_df['score'],
                        y=comments_df['toxicity_score'],
                        mode='markers',
                        name='Comments'
                    ))
                    fig.update_layout(
                        title='Toxicity Score vs. Post/Comment Score',
                        xaxis_title='Score',
                        yaxis_title='Toxicity Score',
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Toxicity Score Distribution")
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=comments_distribution.index,
                        y=comments_distribution.values,
                        name='Comments'
                    ))
                    fig.update_layout(
                        title='Distribution of Toxicity Scores',
                        xaxis_title='Toxicity Score Range',
                        yaxis_title='Number of Comments'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab5:
                st.subheader("Toxicity Trends Over Time")
                
                # Combine posts and comments for trend analysis
                all_content = pd.concat([
                    posts_df[['created_utc', 'toxicity_score', 'type']],
                    comments_df[['created_utc', 'toxicity_score', 'type']]
                ])
                
                # Add year and month columns
                all_content['year'] = all_content['created_utc'].dt.year
                all_content['month'] = all_content['created_utc'].dt.month
                
                # Calculate monthly averages
                monthly_trends = all_content.groupby(['year', 'month', 'type'])['toxicity_score'].mean().reset_index()
                monthly_trends['date'] = pd.to_datetime(monthly_trends[['year', 'month']].assign(day=1))
                
                # Plot monthly trends
                fig = go.Figure()
                for content_type in ['post', 'comment']:
                    type_data = monthly_trends[monthly_trends['type'] == content_type]
                    fig.add_trace(go.Scatter(
                        x=type_data['date'],
                        y=type_data['toxicity_score'],
                        name=f'{content_type.title()}s',
                        mode='lines+markers'
                    ))
                
                fig.update_layout(
                    title='Toxicity Trends Over Time',
                    xaxis_title='Date',
                    yaxis_title='Average Toxicity Score',
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Yearly statistics
                st.subheader("Yearly Toxicity Statistics")
                yearly_stats = all_content.groupby(['year', 'type']).agg({
                    'toxicity_score': ['mean', 'std', 'count']
                }).reset_index()
                
                # Display yearly statistics in a table
                yearly_stats.columns = ['Year', 'Type', 'Mean Toxicity', 'Std Dev', 'Count']
                yearly_stats['Mean Toxicity'] = yearly_stats['Mean Toxicity'].map('{:.1%}'.format)
                yearly_stats['Std Dev'] = yearly_stats['Std Dev'].map('{:.1%}'.format)
                st.dataframe(yearly_stats, use_container_width=True)
                
                # Year-over-year change
                st.subheader("Year-over-Year Change")
                yearly_change = yearly_stats.pivot(index='Year', columns='Type', values='Mean Toxicity')
                yearly_change = yearly_change.pct_change() * 100
                yearly_change = yearly_change.fillna(0)
                
                fig = go.Figure()
                for content_type in ['post', 'comment']:
                    if content_type in yearly_change.columns:
                        fig.add_trace(go.Bar(
                            x=yearly_change.index,
                            y=yearly_change[content_type],
                            name=f'{content_type.title()}s',
                            text=yearly_change[content_type].map('{:.1f}%'.format),
                            textposition='auto',
                        ))
                
                fig.update_layout(
                    title='Year-over-Year Toxicity Change',
                    xaxis_title='Year',
                    yaxis_title='Percentage Change',
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab6:
                st.subheader("Toxic Terms Word Cloud")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Posts Word Cloud")
                    wordcloud_posts = analyzer.get_toxic_word_cloud(posts_df, 'text')
                    if wordcloud_posts is not None:
                        fig1, ax1 = plt.subplots(figsize=(10, 5))
                        ax1.imshow(wordcloud_posts, interpolation='bilinear')
                        ax1.axis('off')
                        st.pyplot(fig1)
                    else:
                        st.info("No toxic posts found in the analyzed content.")
                
                with col2:
                    st.write("Comments Word Cloud")
                    wordcloud_comments = analyzer.get_toxic_word_cloud(comments_df, 'text')
                    if wordcloud_comments is not None:
                        fig2, ax2 = plt.subplots(figsize=(10, 5))
                        ax2.imshow(wordcloud_comments, interpolation='bilinear')
                        ax2.axis('off')
                        st.pyplot(fig2)
                    else:
                        st.info("No toxic comments found in the analyzed content.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check if the subreddit name is correct and try again.")

elif analysis_mode == "Compare Subreddits":
    col1, col2 = st.sidebar.columns(2)
    subreddit1 = col1.text_input("First Subreddit", "python")
    subreddit2 = col2.text_input("Second Subreddit", "programming")
    
    if st.sidebar.button("Compare Subreddits"):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Analyze first subreddit
            status_text.text(f"Analyzing r/{subreddit1}...")
            stats1 = scraper.get_subreddit_stats(subreddit1)
            progress_bar.progress(10)
            
            posts_df1, comments_df1 = scraper.get_subreddit_data(
                subreddit1,
                time_filter=time_filter,
                limit=post_limit
            )
            progress_bar.progress(20)
            
            posts_df1 = analyzer.analyze_dataframe(posts_df1, 'text')
            progress_bar.progress(30)
            
            comments_df1 = analyzer.analyze_dataframe(comments_df1, 'text')
            progress_bar.progress(40)
            
            posts_metrics1 = analyzer.get_toxicity_metrics(posts_df1)
            comments_metrics1 = analyzer.get_toxicity_metrics(comments_df1)
            progress_bar.progress(50)
            
            # Analyze second subreddit
            status_text.text(f"Analyzing r/{subreddit2}...")
            stats2 = scraper.get_subreddit_stats(subreddit2)
            progress_bar.progress(60)
            
            posts_df2, comments_df2 = scraper.get_subreddit_data(
                subreddit2,
                time_filter=time_filter,
                limit=post_limit
            )
            progress_bar.progress(70)
            
            posts_df2 = analyzer.analyze_dataframe(posts_df2, 'text')
            progress_bar.progress(80)
            
            comments_df2 = analyzer.analyze_dataframe(comments_df2, 'text')
            progress_bar.progress(90)
            
            posts_metrics2 = analyzer.get_toxicity_metrics(posts_df2)
            comments_metrics2 = analyzer.get_toxicity_metrics(comments_df2)
            
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            st.success("Comparison completed successfully!")

            # Comparison Header
            st.markdown("---")
            st.markdown(f"<h1 style='text-align: center;'>Comparing r/{subreddit1} vs r/{subreddit2}</h1>", unsafe_allow_html=True)
            st.markdown("---")
            
            # Safety Rating Comparison
            col1, col2 = st.columns(2)
            with col1:
                safety1 = get_safety_rating(
                    (posts_metrics1['mean_toxicity'] + comments_metrics1['mean_toxicity']) / 2,
                    (posts_metrics1['highly_toxic_percentage'] + comments_metrics1['highly_toxic_percentage']) / 2
                )
                st.markdown(f"<h2 style='text-align: center;'>r/{subreddit1}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center;'>{safety1['emoji']} {safety1['rating']}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: {safety1['color']};'>{safety1['description']}</p>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='text-align: center;'>Toxic Score: {safety1['toxic_score']}/100</h4>", unsafe_allow_html=True)
            
            with col2:
                safety2 = get_safety_rating(
                    (posts_metrics2['mean_toxicity'] + comments_metrics2['mean_toxicity']) / 2,
                    (posts_metrics2['highly_toxic_percentage'] + comments_metrics2['highly_toxic_percentage']) / 2
                )
                st.markdown(f"<h2 style='text-align: center;'>r/{subreddit2}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center;'>{safety2['emoji']} {safety2['rating']}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: {safety2['color']};'>{safety2['description']}</p>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='text-align: center;'>Toxic Score: {safety2['toxic_score']}/100</h4>", unsafe_allow_html=True)
            
            # Toxicity Comparison Chart
            st.subheader("Toxicity Comparison")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name=f"r/{subreddit1}",
                x=['Posts', 'Comments'],
                y=[posts_metrics1['mean_toxicity'], comments_metrics1['mean_toxicity']],
                text=[f"{posts_metrics1['mean_toxicity']:.1%}", f"{comments_metrics1['mean_toxicity']:.1%}"],
                textposition='auto',
            ))
            fig.add_trace(go.Bar(
                name=f"r/{subreddit2}",
                x=['Posts', 'Comments'],
                y=[posts_metrics2['mean_toxicity'], comments_metrics2['mean_toxicity']],
                text=[f"{posts_metrics2['mean_toxicity']:.1%}", f"{comments_metrics2['mean_toxicity']:.1%}"],
                textposition='auto',
            ))
            fig.update_layout(
                title='Mean Toxicity Comparison',
                yaxis_title='Toxicity Score',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Community Stats Comparison
            st.subheader("Community Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Subscribers",
                    f"{stats1['subscribers']:,}",
                    f"{stats2['subscribers']:,}"
                )
            with col2:
                st.metric(
                    "Active Users",
                    f"{stats1['active_users']:,}",
                    f"{stats2['active_users']:,}"
                )
            with col3:
                st.metric(
                    "Age (days)",
                    f"{(datetime.now() - stats1['created_utc']).days}",
                    f"{(datetime.now() - stats2['created_utc']).days}"
                )
            
            # Add Word Cloud Comparison
            st.subheader("Toxic Terms Word Cloud Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"r/{subreddit1} Word Cloud")
                wordcloud1 = analyzer.get_toxic_word_cloud(comments_df1, 'text')
                if wordcloud1 is not None:
                    fig1, ax1 = plt.subplots(figsize=(10, 5))
                    ax1.imshow(wordcloud1, interpolation='bilinear')
                    ax1.axis('off')
                    st.pyplot(fig1)
                else:
                    st.info(f"No toxic content found in r/{subreddit1}.")
            
            with col2:
                st.write(f"r/{subreddit2} Word Cloud")
                wordcloud2 = analyzer.get_toxic_word_cloud(comments_df2, 'text')
                if wordcloud2 is not None:
                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    ax2.imshow(wordcloud2, interpolation='bilinear')
                    ax2.axis('off')
                    st.pyplot(fig2)
                else:
                    st.info(f"No toxic content found in r/{subreddit2}.")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check if the subreddit names are correct and try again.")

else:
    st.info("Select an analysis mode and enter subreddit name(s) to begin analysis.") 