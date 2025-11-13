import streamlit as st
import pandas as pd
from openai import OpenAI
import os

# ================================
# Load Dataset
# ================================
df = pd.read_csv("movie_cleaned.csv")

# Only preview for debugging
st.write("Dataset loaded:")
st.write(df.head(5))

# ================================
# Streamlit UI
# ================================
st.title("🎬 Weebsu — Movie Recommender")
st.write("Ask for recommendations based on any movie you like!")

# User inputs API key (NOT saved, NOT written to disk)
api_key = st.text_input(
    "🔑 Enter your OpenAI API Key",
    type="password",
    help="Your key is not saved or stored. It is only used for this session."
)

# User enters movie question
question = st.text_area("🎤 What do you want to ask Weebsu?", height=120)

# Button to send query
ask_button = st.button("💬 Get Recommendation")

# ================================
# When pressed
# ================================
if ask_button:

    if api_key.strip() == "":
        st.error("⚠ Please enter your API key first.")
    elif question.strip() == "":
        st.error("⚠ Please enter a question.")
    else:
        # Set environment variable only temporarily
        os.environ["OPENAI_API_KEY"] = api_key.strip()

        # Create client
        client = OpenAI()

        # Convert a safe subset of data to string
        df_string = df.head(50).to_string()

        # System prompt
        system_prompt = (
            "You are a professional data analyst and a film critic with deep knowledge of cinema history, "
            "genres, filmmaking styles, visual aesthetics, narrative tropes, and critical reception patterns. "
            "You must use the provided dataset strictly as your analytical base. If the user asks something "
            "outside of this dataset, answer that the data is not available. "
            "When given the dataset (`df_string`), treat it as your entire working table and perform "
            "Python-like reasoning using filtering, grouping, and similarity analysis.\n\n"
            "Your main function is to act as a cinephile-style recommendation engine: "
            "the user gives a movie they like, and you must recommend at least two movies from the dataset based on:\n"
            "• genre similarity\n"
            "• director or production company connections\n"
            "• narrative themes\n"
            "• tone and mood\n"
            "• critic reception and popularity\n\n"
            "Each recommendation must include a cinematic explanation of why it fits the user's taste.\n\n"
            "Here is the dataset:\n" + df_string
        )

        # Make the API call
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ]
            )

            answer = response.choices[0].message.content
            st.markdown("### 🎞️ Weebsu's Answer:")
            st.write(answer)

        except Exception as e:
            st.error(f"Error: {e}")
