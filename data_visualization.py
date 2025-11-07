import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data():
    # Load processed data
    try:
        df = pd.read_csv('processed_data.csv')
        print("Processed data loaded for visualization.")
        print(f"Dataset shape: {df.shape}")
    except FileNotFoundError:
        print("Error: 'processed_data.csv' not found. Run 'data_preprocessing.py' first.")
        return

    # Set style for better plots
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1. Distribution of Ratings
    plt.figure()
    sns.countplot(x='Score', data=df, palette='viridis')
    plt.title('Distribution of Product Ratings')
    plt.xlabel('Rating Score')
    plt.ylabel('Number of Reviews')
    plt.tight_layout()
    plt.show()

    # 2. Distribution of Helpfulness Ratio
    plt.figure()
    sns.histplot(df['HelpfulnessRatio'], bins=20, kde=True, color='purple', alpha=0.7)
    plt.title('Distribution of Helpfulness Ratio')
    plt.xlabel('Helpfulness Ratio')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # 3. Average Rating Over Years
    avg_rating_year = df.groupby('review_year')['Score'].mean().reset_index()
    plt.figure()
    sns.lineplot(x='review_year', y='Score', data=avg_rating_year, marker='o', color='green')
    plt.title('Average Rating Over Years')
    plt.xlabel('Year')
    plt.ylabel('Average Rating')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4. Top 10 Products by Number of Reviews
    top_products = df['ProductId'].value_counts().head(10)
    plt.figure()
    sns.barplot(x=top_products.values, y=top_products.index, palette='cubehelix')
    plt.title('Top 10 Products by Number of Reviews')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Product ID')
    plt.tight_layout()
    plt.show()

    # 5. Top 10 Users by Number of Reviews (corrected: 'User Id' no space)
    top_users = df['UserId'].value_counts().head(10)
    plt.figure()
    sns.barplot(x=top_users.values, y=top_users.index, palette='rocket')
    plt.title('Top 10 Users by Number of Reviews')
    plt.xlabel('Number of Reviews')
    plt.ylabel('User  ID')
    plt.tight_layout()
    plt.show()

    print("All visualizations generated successfully.")

if __name__ == "__main__":
    visualize_data()