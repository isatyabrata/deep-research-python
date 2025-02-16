# deep-research-python/deep_research_lib/run.py
import asyncio
import sys

from deep_research_lib.deep_research import deep_research, write_final_report
from deep_research_lib.feedback import generate_feedback
from deep_research_lib.output_manager import OutputManager

output = OutputManager()

# Helper function for consistent logging
def log(*args):
    output.log(*args)

# Helper function to get user input (using asyncio for async compatibility if needed later)
async def ask_question(query):
    print(query, end='', flush=True) # Print query without newline and flush
    return await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)

async def run_research():
    # Get initial query
    initial_query = (await ask_question('What would you like to research? ')).strip()

    # Get breadth and depth parameters
    while True:
        breadth_str = (await ask_question('Enter research breadth (recommended 2-10, default 4): ')).strip()
        try:
            breadth = int(breadth_str) if breadth_str else 4
            break
        except ValueError:
            log("Invalid breadth. Please enter a number.")

    while True:
        depth_str = (await ask_question('Enter research depth (recommended 1-5, default 2): ')).strip()
        try:
            depth = int(depth_str) if depth_str else 2
            break
        except ValueError:
            log("Invalid depth. Please enter a number.")

    log("Creating research plan...")

    # Generate follow-up questions
    follow_up_questions = await generate_feedback(query=initial_query)

    log("\nTo better understand your research needs, please answer these follow-up questions:")

    # Collect answers to follow-up questions
    answers = []
    for question in follow_up_questions:
        answer = (await ask_question(f"\n{question}\nYour answer: ")).strip()
        answers.append(answer)

    # Combine all information for deep research
    combined_query = f"""
Initial Query: {initial_query}
Follow-up Questions and Answers:
{chr(10).join([f'Q: {q}{chr(10)}A: {answers[i]}' for i, q in enumerate(follow_up_questions)])}
    """.strip() # using chr(10) for newline to be explicit

    log("\nResearching your topic...")
    log("\nStarting research with progress tracking...\n")

    async def progress_callback(progress): # Define callback function inside run_research
        output.update_progress(progress)

    learnings, visited_urls = await deep_research(
        query=combined_query,
        breadth=breadth,
        depth=depth,
        on_progress=progress_callback # Pass the callback function
    )
    # Flatten the learnings list if it's nested
    flat_learnings = []
    for item in learnings:
        if isinstance(item, list): # Check if item is a list (nested list)
            flat_learnings.extend(item) # Extend with elements of the inner list
        else:
            flat_learnings.append(item) # Append if it's already a string
    log("Writing final report...")
    print(f"\n--- Learnings before join ---\n{learnings}\n--- End Learnings ---\n") # Debug print
    print(f"Type of 'learnings': {type(learnings)}") # Print type
    print(f"Content of 'learnings': {learnings}") # Print content (again)
    print(f"Is 'learnings' a list?: {isinstance(learnings, list)}") # Check if it's a list
    print(f"Length of 'learnings': {len(learnings)}") # Print length
    print(f"--- End Learnings ---\n")
    print(f"\n\nLearnings:\n\n{chr(10).join(learnings)}") # Use print() instead of log()
    # log(f"\n\nLearnings:\n\n{chr(10).join(learnings)}") # using chr(10) for newline
    log(f"\n\nVisited URLs ({len(visited_urls)}):\n\n{chr(10).join(visited_urls)}") # using chr(10) for newline
    # log("Writing final report...")

    report_markdown = await write_final_report(
        prompt=combined_query,
        learnings=learnings,
        visited_urls=visited_urls
    )

    # Save report to file
    with open('output.md', 'w', encoding='utf-8') as f:
        f.write(report_markdown)

    print(f"\n\nFinal Report:\n\n{report_markdown}")
    print('\nReport has been saved to output.md')

if __name__ == "__main__":
    asyncio.run(run_research())