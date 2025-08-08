import streamlit as st
import pandas as pd
from openai import OpenAI

# Load OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# Title
st.title("LOOP AI Assistant 🤖")
st.markdown("Upload your demand prediction CSV and ask questions about it!")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.subheader("📊 Preview")
        st.dataframe(df.head())

        # User question
        user_question = st.text_input("💬 Ask a question about your data:")

        if user_question:
            with st.spinner("Thinking... 💭"):
                df_sample = df.head(50).to_string(index=False)

                prompt = f"""
You are a helpful data analyst. Here's a sample of ride demand data:
{df_sample}

Based on this, answer the question:
{user_question}
                """

                # Make chat completion call
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful and concise data analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )

                answer = response.choices[0].message.content
                st.subheader("🧠 Answer")
                st.write(answer)

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("Please upload a CSV file to begin.")
