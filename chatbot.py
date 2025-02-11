import streamlit as st
from transformers import pipeline

# Load a pre-trained Hugging Face model for medical question answering
chatbot = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Define a healthcare-specific response function
def healthcare_chatbot(user_input):
    # Predefined responses for specific keywords
    keywords = {
        "symptom": "It seems like you're experiencing symptoms. Please consult a doctor for accurate advice.",
        "appointment": "Would you like me to schedule an appointment with a doctor?",
        "medication": "It's important to take your prescribed medications regularly. If you have concerns, consult your doctor.",
        "emergency": "If this is a medical emergency, please call an ambulance or visit the nearest hospital immediately."
    }
    
    for key, response in keywords.items():
        if key in user_input.lower():
            return response

    # Provide a medical context for better responses
    medical_context = """
    I am a healthcare assistant trained to provide general medical guidance. However, I am not a doctor.
    Always consult a medical professional for serious health concerns.
    """

    # Use Hugging Face model for question-answering
    model_response = chatbot(question=user_input, context=medical_context)
    
    # Ensure meaningful responses
    if model_response["score"] > 0.5:  # Ensures confidence in the answer
        return model_response["answer"]
    else:
        return "I'm sorry, I couldn't find a confident answer. Please consult a medical professional."

# Streamlit web app interface
def main():
    st.title("🩺 AI Healthcare Assistant Chatbot")

    # User input
    user_input = st.text_input("How can I assist you today?", "")

    if st.button("Submit"):
        if user_input:
            st.write("**User:**", user_input)
            response = healthcare_chatbot(user_input)
            st.write("**Healthcare Assistant:**", response)
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()
