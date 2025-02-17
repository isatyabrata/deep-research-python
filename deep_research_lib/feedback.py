# deep-research-python/deep_research_lib/feedback.py
from pydantic import BaseModel, Field
from typing import List, Union

from deep_research_lib.ai.providers import o3_mini_model
from deep_research_lib.prompt import system_prompt
from deep_research_lib.ai.providers import llm_provider # Assuming you will create openai_provider.py later

format = """{
  "questions": ["Question 1", "Question 2", "Question 3"....]
}"""

class FeedbackResponse(BaseModel):
    questions: List[str] = Field(alias='questions', description="Follow up questions to clarify the research direction, max of numQuestions") # Added alias for '_questions'

async def generate_feedback(query: str, num_questions: int = 3) -> List[str]:
    user_feedback = await llm_provider.generate_object(
        prompt=f"Given the following query from the user, ask some follow up questions to clarify the research direction. Return a minimum of {num_questions} questions, but feel free to return less if the original query is clear: <query>{query}</query>, Output should be in format: {format}",
        system=system_prompt(),
        response_model=FeedbackResponse # Using Pydantic model for schema
    )
    return user_feedback.questions # Slicing list like in typescript