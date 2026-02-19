from groq import Groq
import streamlit as st
from typing import List


class GroqQA:
    def __init__(self):
        self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    def generate_answer(self, question: str, context: List[str]) -> str:
        if not context:
            return "Not enough information to answer."

        context_text = "\n\n".join(context)

        prompt = f"""
Use the context below to answer the question.

Context:
{context_text}

Question:
{question}

Answer:
"""

        completion = self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return completion.choices[0].message.content
