import spacy
import re
from textblob import TextBlob
from flask import Flask, request, jsonify

# Load the spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Initialize Flask app
app = Flask(__name__)

class HospitalChatbot:
    def __init__(self):
        # Dictionary of common health issues and corresponding labels
        self.health_issues = {
            "fever": "Infection-related issues",
            "cough": "Respiratory issues",
            "headache": "Neurological or stress-related issues",
            "chest pain": "Cardiac issues",
            "nausea": "Digestive issues",
            "fatigue": "General symptoms of illness",
            "joint pain": "Musculoskeletal issues",
            "dizziness": "Balance and neurological issues",
            "shortness of breath": "Respiratory/Cardiac issues",
            "rash": "Dermatological issues",
            "swelling": "Inflammation-related issues"
        }

    def get_health_issue_label(self, symptom):
        """
        This function will check if the symptom matches known health issues
        and return the relevant label.
        Uses spaCy for processing and re for matching complex patterns.
        """
        # Process the input text using spaCy
        doc = nlp(symptom.lower())
        symptoms = [token.text for token in doc if token.pos_ == "NOUN"]
        
        for issue in self.health_issues:
            if any(issue in symptom for symptom in symptoms):
                return self.health_issues[issue]
        return "General health consultation"

    def provide_response(self, user_input):
        """
        Provide an appropriate response based on user input (symptoms).
        """
        label = self.get_health_issue_label(user_input)
        
        # Provide responses based on the label
        responses = {
            "Infection-related issues": "You may have an infection. Please consult a doctor for further diagnosis. You might need tests like blood work.",
            "Respiratory issues": "It seems like a respiratory issue. If you're experiencing shortness of breath or persistent cough, please see a pulmonologist.",
            "Neurological or stress-related issues": "You might be dealing with stress or a neurological issue. A consultation with a neurologist or psychologist could help.",
            "Cardiac issues": "Chest pain or shortness of breath could indicate a heart condition. Please consult a cardiologist immediately.",
            "Digestive issues": "Nausea and stomach-related issues may require a gastroenterologist's assessment. Please share more details about your symptoms.",
            "General symptoms of illness": "Fatigue can be related to a number of conditions. Please monitor your health and consult a general physician if necessary.",
            "Musculoskeletal issues": "Joint pain or muscle discomfort may require an orthopedic or physiotherapy consultation.",
            "Balance and neurological issues": "Dizziness could be related to a number of causes, including neurological issues. Please consult a specialist.",
            "Dermatological issues": "Rashes or skin conditions may require a dermatologist consultation. Do you have any visible rashes?",
            "Inflammation-related issues": "Swelling can be caused by inflammation. Please share more about the affected area and any additional symptoms.",
            "General health consultation": "It looks like you're describing a general health issue. Please consult a doctor to assess your symptoms."
        }

        return responses.get(label, "I'm not sure about that. Could you please provide more details?")

    def sentiment_analysis(self, user_input):
        """
        This function uses TextBlob to analyze the sentiment of the input and adjust the response accordingly.
        """
        blob = TextBlob(user_input)
        sentiment = blob.sentiment.polarity
        if sentiment < 0:
            return "I'm sorry you're feeling unwell. Let me assist you in the best way possible."
        elif sentiment > 0:
            return "It's great to hear that you're feeling better! Let me know if there's anything I can help you with."
        else:
            return "I understand. Let me help you with your symptoms."

    def start_chat(self):
        """
        Function to start the conversation with the patient.
        This is for CLI-based interactions.
        """
        print("Welcome to the hospital reception chatbot!")
        print("Please describe your symptoms to assist in directing you to the right department.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "stop"]:
                print("Thank you for using the hospital chatbot. Stay healthy!")
                break
            sentiment_response = self.sentiment_analysis(user_input)
            response = self.provide_response(user_input)
            print(f"Chatbot: {sentiment_response} {response}")


# Initialize the chatbot
chatbot = HospitalChatbot()

@app.route('/chat', methods=['POST'])
def chat():
    """
    This function handles the chatbot interaction in a web-based interface using Flask.
    """
    data = request.get_json()
    user_input = data.get("message", "")
    
    if not user_input:
        return jsonify({"response": "Please provide symptoms for assistance."})
    
    sentiment_response = chatbot.sentiment_analysis(user_input)
    response = chatbot.provide_response(user_input)
    
    return jsonify({"response": f"{sentiment_response} {response}"})


# Start the Flask app (For web-based interaction)
if __name__ == '__main__':
    app.run(debug=True)

