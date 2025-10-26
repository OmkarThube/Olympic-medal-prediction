document.getElementById('predictionForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the default form submission

    const country = document.getElementById('country').value;

    // Make a POST request to the server with the country data in JSON format
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',  // Set content type to JSON
        },
        body: JSON.stringify({ country: country })  // Send country data as JSON
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(errData => { throw new Error(errData.error); });
        }
        return response.json();
    })
    .then(data => {
        const resultContent = document.getElementById('result-content');
        resultContent.innerHTML = '';

        if (data.results) {
            // Extract and display Summer and Winter predictions
            const summerLogistic = data.results.summer.logistic_prediction;
            const summerRandomForest = data.results.summer.random_forest_prediction;
            const winterLogistic = data.results.winter.logistic_prediction;
            const winterRandomForest = data.results.winter.random_forest_prediction;

            const conclusion = data.conclusion;

            resultContent.innerHTML = `
                <p><strong>Country:</strong> ${country}</p>
                <p><strong>Summer Olympics:</strong></p>
                <ul>
                    <li>Logistic Regression Prediction: ${summerLogistic}</li>
                    <li>Random Forest Prediction: ${summerRandomForest}</li>
                </ul>
                <p><strong>Winter Olympics:</strong></p>
                <ul>
                    <li>Logistic Regression Prediction: ${winterLogistic}</li>
                    <li>Random Forest Prediction: ${winterRandomForest}</li>
                </ul>
                <p><strong>Best Model:</strong></p>
                <ul>
                    <li>Summer: ${conclusion.summer_best_algorithm}</li>
                    <li>Winter: ${conclusion.winter_best_algorithm}</li>
                </ul>
            `;

            // Show histograms for predictions
            createHistogram('Summer Olympics', summerLogistic, summerRandomForest);
            createHistogram('Winter Olympics', winterLogistic, winterRandomForest);
        } else {
            resultContent.innerHTML = '<p>No data available for the chosen country.</p>';
        }
    })
    .catch(error => {
        // Display error message from the server
        document.getElementById('result-content').innerHTML = `<p>Error: ${error.message}</p>`;
        document.getElementById('chartSection').style.display = 'none';  // Hide chart on error
    });
});