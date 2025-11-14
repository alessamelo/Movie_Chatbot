import streamlit as st
from openai import OpenAI
import pandas as pd

# ================================
# ✅ Load OpenAI API key securely
# ================================
api_key = st.secrets["OPENAI_API_KEY"]  # You must set this in Streamlit Cloud or secrets.toml
client = OpenAI(api_key=api_key)

# ================================
# Load Dataset
# ================================
df = pd.read_csv("movie_cleaned.csv")
df_subset = df.head(50)  # Limit preview to avoid token overload
df_string = df_subset.to_string()

# ================================
# Streamlit UI
# ================================
st.title("🎬 Weebsu — Movie Recommender")
st.write("Ask for recommendations based on your favorite movies! 🍿")

# User enters question
user_input = st.text_area("🎤 What do you want to ask Weebsu?", height=120)

# ================================
# When user submits question
# ================================
if user_input:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional data analyst and a film critic with deep knowledge of cinema history, "
                        "genres, filmmaking styles, visual aesthetics, narrative tropes, and critical reception patterns. "
                        "Use ONLY the provided dataset as your base. "
                        "If the user asks something outside this dataset, respond with: "
                        "'Sorry, that information is not available in the current dataset.'\n\n"
                        "When given the dataset (`df_string`), treat it as your entire working table and perform "
                        "Python-like reasoning using filtering, grouping, and similarity analysis.\n\n"
                        "Your main goal is to act as a cinephile-style recommendation engine: "
                        "the user provides a movie they like, and you must recommend at least two movies from the dataset "
                        "based on:\n"
                        "• Genre similarity\n"
                        "• Director or production company\n"
                        "• Narrative themes and tone\n"
                        "• Critic reception and popularity\n\n"
                        "Each recommendation must include a short cinematic explanation of why it fits the user's taste.\n\n"
                        "Here is the dataset:\n" + df_string
                    )
                },
                {"role": "user", "content": user_input}
            ]
        )

        # Display response
        answer = response.choices[0].message.content
        st.subheader("🎞️ Weebsu's Answer:")
        st.write(answer)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

