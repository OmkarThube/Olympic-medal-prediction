from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)


# Step 1: Load dataset (hardcoded in the code)
def load_data():
    filepath = r"C:\Users\omk57\Downloads\olympics_dataset.csv"
    return pd.read_csv(filepath)


# Step 2: Preprocessing - Create binary 'Medal_Won' column and aggregate data
def preprocess_data(data):
    data['Medal_Won'] = data['Medals'].apply(lambda x: 0 if x == 'No medal' else 1)
    team_year_data = data.groupby(['Team', 'Year', 'Sport']).agg(
        total_participation=('ID', 'count'),
        medals_won=('Medal_Won', 'sum')
    ).reset_index()

    # Calculate win percentage
    team_year_data['win_percentage'] = (team_year_data['medals_won'] / team_year_data['total_participation']) * 100
    return team_year_data


# Function to plot bar chart with country names and win percentages
def plot_histogram(sport_data, sport):
    plt.figure(figsize=(12, 6))

    # Filter data for only those countries with some winning chances
    filtered_data = sport_data[sport_data['win_percentage'] > 0]

    # Sort data by win percentage
    sorted_data = filtered_data.sort_values(by='win_percentage', ascending=False)

    # Plot bar chart
    sns.barplot(x='Team', y='win_percentage', data=sorted_data, palette='coolwarm')

    # Add titles and labels
    plt.title(f'Win Percentage by Country for {sport}', fontsize=16)
    plt.xlabel('Country (Team)', fontsize=12)
    plt.ylabel('Win Percentage', fontsize=12)

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Save the chart as an image to display in the HTML
    image_path = os.path.join('static', 'histogram.png')
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        sport = request.form.get("sport")
        data = load_data()
        team_year_data = preprocess_data(data)

        sport_data = team_year_data[team_year_data['Sport'] == sport]
        if sport_data.empty:
            return render_template('index.html', error=f"No data available for {sport}.")

        # Features and labels for both models
        X = sport_data[['total_participation', 'medals_won']]

        # Binarize win_percentage (e.g., 50% threshold)
        sport_data['win_success'] = sport_data['win_percentage'].apply(lambda x: 1 if x > 50 else 0)
        y = sport_data['win_success']

        # Split data for training/testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Naive Bayes Model
        naive_bayes = GaussianNB()
        naive_bayes.fit(X_train, y_train)
        y_pred_naive = naive_bayes.predict(X_test)
        naive_accuracy = accuracy_score(y_test, y_pred_naive)

        # Logistic Regression Model
        logistic_reg = LogisticRegression(max_iter=1000)
        logistic_reg.fit(X_train, y_train)
        y_pred_logistic = logistic_reg.predict(X_test)
        logistic_accuracy = accuracy_score(y_test, y_pred_logistic)

        # Calculate predicted chances of winning for both models
        sport_data['winning_chance_naive_bayes'] = naive_bayes.predict_proba(X)[:, 1] * 100
        sport_data['winning_chance_logistic'] = logistic_reg.predict_proba(X)[:, 1] * 100

        # Filter only countries with winning chances > 0
        sport_data = sport_data[(sport_data['winning_chance_naive_bayes'] > 0) |
                                (sport_data['winning_chance_logistic'] > 0)]

        if sport_data.empty:
            return render_template('index.html', error=f"No countries have a chance of winning in {sport}.")
        
        results_df = pd.DataFrame({
            'Team': sport_data['Team'],
        'Winning_Probability_NB': sport_data['winning_chance_naive_bayes'],
        'Winning_Probability_LR': sport_data['winning_chance_logistic']
        })
        results_df.to_csv('winning_probabilities.csv', index=False)


        # Generate histogram and save it
        plot_histogram(sport_data, sport)

        # Decide which model is more accurate
        more_accurate = "Logistic Regression" if logistic_accuracy > naive_accuracy else "Naive Bayes"

        # Send data to the HTML page
        results = sport_data[['Team', 'total_participation', 'medals_won', 'win_percentage']].to_dict(orient='records')
        for i, result in enumerate(results):
            result['winning_chance_logistic'] = round(sport_data['winning_chance_logistic'].iloc[i], 2)
            result['winning_chance_naive_bayes'] = round(sport_data['winning_chance_naive_bayes'].iloc[i], 2)

        return render_template('index.html', sport=sport, results=results, logistic_accuracy=logistic_accuracy * 100,
                               naive_accuracy=naive_accuracy * 100, more_accurate=more_accurate)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
