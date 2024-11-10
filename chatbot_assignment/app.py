'''
BIA 6304 - Final Project - Azmina Sayyed

This is the application to initiate the chat session. It is a flask based web application that retrieves
information from the loaded dataset but also uses GenAI's gemini flah model to curate responses for the users.

'''
# Importing the necessary libraries
from flask import Flask, request, jsonify, session
import pandas as pd
import os
import numpy as np
from .dqm import DocumentQueryModel
from .embedding import HuggingFaceEmbedding
import google.generativeai as genai  # Import Google Generative AI library
from flask_cors import CORS

# Set yourFlask app & secret key here
app = Flask(__name__)
CORS(app)
app.secret_key = ' '  # Set your secret key here

# Initialize the Google Generative AI model
genai_api_key = " "  # Enter your genai API key here.
genai.configure(api_key=genai_api_key)

# Initialize embedding function with HuggingFace model
embedding_function = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create the DocumentQueryModel instance
dqm_instance = DocumentQueryModel(
    data=DocumentQueryModel.new(),  # Starting with an empty DataFrame
    db_path=os.path.join("assets", "space_dataset.pkl"),  # Path for the Database
    embedding_function=embedding_function
)

def load_jsonl(file_path: str, id_key: str, content_key: str) -> None:
    """
    Load data from a JSONL file into the DocumentQueryModel.
    """
    dqm_instance.load_jsonl(file_path, id_key, content_key)

# Load documents
load_jsonl(
    file_path=os.path.join("assets", "space_doc.jsonl"),  # Adjust the path as necessary
    id_key="title",
    content_key="document"
)
print(f"Loaded documents count: {dqm_instance.document_count}")

@app.route('/ask', methods=['POST'])
async def ask():
    user_input = request.json['input']
    print(f"User Input: {user_input}")

# Return relevant documents from the dataset.
    dqm_results = dqm_instance.query(user_input, top_n=5)  
    print(dqm_results)

# Store chat history
    if 'conversation_history' not in session:
        session['conversation_history'] = []

    if not dqm_results.empty:
        assistant_content = "Here is some information that might help:\n"
        assistant_content += "\n".join(dqm_results['content'].values)
        
       # Prepare the message for the chat model
        full_prompt = f"{assistant_content}\n\nUser Query: {user_input}"
        model= genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate response using Google Generative AI
        ai_response = model.generate_content(full_prompt)
        print("AI Response:", ai_response)

            
        # Inspect the attributes of ai_response
        # print("Attributes of ai_response:", dir(ai_response))
        
        # Check if ai_response is a GenerateContentResponse object
        if isinstance(ai_response, genai.types.generation_types.GenerateContentResponse):
            # Access the genai response content
            try:
                # Print the raw ai_response to understand its structure
                # print("Raw AI Response:", ai_response)
                
                # Access candidates or other relevant attributes based on your inspection
                candidates = ai_response.candidates
                if candidates and len(candidates) > 0:
                    response_text = candidates[0].content.parts[0].text
                    response_text = response_text.replace('\n', '')  # Strip leading/trailing whitespace
                      # Store the conversation in history
                    session['conversation_history'].append({
                    "user_input": user_input,
                    "assistant_response": response_text
                    })

                    return jsonify({"Assistant": response_text})
                   
                else:
                    return jsonify({"Assistant": "No candidates found in the AI response."})
            except Exception as e:
                return jsonify({"Assistant": "Error extracting response text: " + str(e)})
        else:
            return jsonify({"Assistant": "AI response is not in the expected format."})
    else:
        return jsonify({"Assistant": "Sorry, I couldn't find any relevant information."})
      
if __name__ == '__main__':
    app.run(debug=True)
