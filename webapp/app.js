// This is a javascript file to run the web app on the flask server using local host..
document.getElementById('queryForm').addEventListener('submit', async function(event) {
    event.preventDefault(); // Prevent form from submitting normally
    
    const userInput = document.getElementById('userInput').value;
    const responseDiv = document.getElementById('response');

    try {
        const response = await fetch('http://127.0.0.1:5000/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ input: userInput })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        let cleanedResponse = data.Assistant.replace(/\*\*\*+/g, '').trim();
        responseDiv.innerText = `Assistant: ${cleanedResponse}`;
        
    } catch (error) {
        responseDiv.innerText = 'Error: ' + error.message;
    }
});
