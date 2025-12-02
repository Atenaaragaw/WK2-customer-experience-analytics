import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Configure plotting style for professional look
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans']

# --- REAL QUERY 1 DATA: Comparative Sentiment Analysis (CONFIRMED) ---
comparative_df = pd.DataFrame({
    'bank_name': ["Bank of Abyssinia", "Commercial Bank of Ethiopia", "Dashen Bank"],
    'total_reviews': [352, 325, 337],
    'average_rating': [3.20, 3.98, 3.82],
    'percent_positive': [39.77, 55.69, 57.27]
})
# --- THEMATIC DATA: Confirmed Dashen themes + MOCK data for CBE/BOA ---
# Note: For accurate final report, full Query 2 & 3 data is needed.
full_theme_text = {
    # Mocking for CBE: High rating implies good core functionality but some bugs.
    'CBE': "Security, Fast Login, Reliable Transfers, App Stability Issues, Customer Service Wait Times, System Downtime, Strong Support",
    # Confirmed Dashen volatility (Transaction Performance and UI/UX drive both positive and negative sentiment)
    'Dashen': "Transaction Performance, User Interface (UI) & Experience, Slow App, Bugs, Account Access, Excellent Interface, Quick Transfers", 
    # Mocking for BOA: Low performance implies major problems like crashes.
    'BOA': "App Crashes, Login Errors, Slow Processing, OTP Delays, Customer Support Problems, Limited Features, Frequent Errors"
}
# --- END THEMATIC DATA ---

def create_comparative_bar_chart(df):
    """Visualizes Average Rating and Percent Positive for bank comparison."""
    # Ensure correct bank order for plotting (by Positive Sentiment)
    df_sorted = df.sort_values(by='percent_positive', ascending=False)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart for Average Rating
    sns.barplot(x='bank_name', y='average_rating', data=df_sorted, ax=ax1, color='#009688', label='Average Rating')
    ax1.set_ylim(0, 5)
    ax1.set_ylabel('Average Rating (1-5)', color='#009688')
    ax1.tick_params(axis='y', labelcolor='#009688')
    
    # Line chart for Percent Positive (on secondary axis)
    ax2 = ax1.twinx()
    sns.lineplot(x='bank_name', y='percent_positive', data=df_sorted, ax=ax2, color='#FF5722', marker='o', linewidth=2, label='% Positive Reviews')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Percent Positive Reviews (%)', color='#FF5722')
    ax2.tick_params(axis='y', labelcolor='#FF5722')
    
    fig.suptitle('Comparative Performance: Rating and Sentiment by Bank', fontsize=16)
    ax1.set_xlabel('Bank')
    ax1.grid(axis='y', linestyle='--')
    fig.tight_layout()
    plt.show()

def create_sentiment_trend_chart(df):
    """Simulates and visualizes sentiment trend over time (Required Plot 2)."""
    # Create mock date range (12 months)
    start_date = pd.to_datetime('2024-01-01')
    end_date = pd.to_datetime('2024-12-01')
    monthly_periods = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # Simulate trend for each bank based on current positive rate (higher rate means better trend stability)
    data = []
    for index, row in df.iterrows():
        bank_name = row['bank_name']
        baseline_rate = row['percent_positive'] / 100 
        
        # Introduce slight monthly noise, trending slightly downwards (common in apps)
        trend = [(baseline_rate + (i * -0.01) + (0.02 * (i % 3))) for i in range(len(monthly_periods))]
        
        for date, rate in zip(monthly_periods, trend):
            data.append({
                'Date': date,
                'Bank': bank_name,
                'Positive Rate (%)': min(max(rate * 100, 30), 80) # Keep rate between 30% and 80%
            })

    df_trend = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Date', y='Positive Rate (%)', hue='Bank', data=df_trend, marker='o', linewidth=2)
    plt.title('Simulated Monthly Positive Sentiment Trend (Proxy for Stability)', fontsize=16)
    plt.xlabel('Month')
    plt.ylabel('Positive Sentiment Rate (%)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(title='Bank')
    plt.tight_layout()
    plt.show()

def create_bank_word_clouds(themes_dict):
    """Generates a word cloud for themes across all banks (Required Plot 3)."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    banks = ['CBE', 'Dashen', 'BOA'] # Use simplified keys for plotting

    for i, bank_key in enumerate(banks):
        text = themes_dict[bank_key]
        
        # Word cloud generation parameters
        wordcloud = WordCloud(
            width=800, height=400, background_color='white', 
            colormap='viridis', max_words=15, 
            stopwords=set(['and', 'the', 'bank', 'app', 'is', 'a', 'to', 'for', 'of', 'in', 'issues']) 
        ).generate(text)
        
        # Determine the full name for the title
        full_name = comparative_df[comparative_df['bank_name'].str.contains(bank_key)]['bank_name'].iloc[0]
        
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].set_title(f'Top Themes: {full_name}', fontsize=14)
        axes[i].axis("off")
        
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. Comparative Performance Plot
    create_comparative_bar_chart(comparative_df)
    
    # 2. Sentiment Trend Plot (Simulated)
    create_sentiment_trend_chart(comparative_df) 
    
    # 3. Thematic Word Clouds (Using confirmed Dashen data and mock CBE/BOA data)
    create_bank_word_clouds(full_theme_text) 

    print("Visualization generation script executed. Mock data was used for CBE and BOA themes. Please review the generated plots.")