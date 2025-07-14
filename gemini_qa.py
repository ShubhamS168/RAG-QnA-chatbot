import google.generativeai as genai
from typing import List

from config import Config

class GeminiQA:
    """
    Handles interaction with the Google Gemini model for question answering.
    """
    def __init__(self):
        """
        Initializes the Gemini API with the provided API key.
        """
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            # Initialize the generative model
            # self.model = genai.GenerativeModel('gemini-pro')
            self.model = genai.GenerativeModel('gemini-2.0-flash')
        except Exception as e:
            raise RuntimeError(f"Failed to configure Gemini API or load model: {e}")

    def generate_answer(self, question: str, context: List[str]) -> str:
        """
        Generates an answer using the Gemini model based on the provided question and context.

        Args:
            question (str): The user's question.
            context (List[str]): A list of retrieved text chunks relevant to the question.

        Returns:
            str: The AI-generated answer.
        """
        if not context:
            return "I don't have enough information in the provided documents to answer that question. Please upload relevant documents."

        # Join the context chunks into a single string
        context_str = "\n\n".join(context)

        # Define the prompt template
        prompt_template = """You are an intelligent assistant. Use the context below to answer the question accurately.
        Context: {{context}}
        Question: {{question}}
        Answer:"""

        # Populate the prompt template with actual context and question
        full_prompt = prompt_template.replace("{{context}}", context_str).replace("{{question}}", question)

        try:
            # Generate content using the Gemini model
            response = self.model.generate_content(full_prompt)
            # Access the text from the response
            return response.text
        except Exception as e:
            print(f"Error generating answer with Gemini: {e}")
            return "An error occurred while generating the answer. Please try again."

# Example usage (for testing purposes, not part of the main app flow):
if __name__ == "__main__":
    # This block will only run if GEMINI_API_KEY is set in .env
    # For a real test, ensure your .env is configured.
    try:
        qa_system = GeminiQA()
        sample_context = [
            "The capital of France is Paris.",
            "Paris is known for its Eiffel Tower and Louvre Museum.",
            "The river Seine flows through Paris."
        ]
        sample_question = "What is the capital of France and what is it known for?"
        answer = qa_system.generate_answer(sample_question, sample_context)
        print(f"Question: {sample_question}")
        print(f"Answer: {answer}")

        sample_question_no_context = "What is the highest mountain in the world?"
        answer_no_context = qa_system.generate_answer(sample_question_no_context, [])
        print(f"\nQuestion: {sample_question_no_context}")
        print(f"Answer: {answer_no_context}")

    except RuntimeError as e:
        print(f"Initialization error: {e}")
        print("Please ensure your GEMINI_API_KEY is correctly set in the .env file.")
