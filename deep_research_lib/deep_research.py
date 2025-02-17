# deep-research-python/deep_research_lib/deep_research.py
import asyncio
from typing import List, Optional, Dict, Any, Callable, Set
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor # For concurrent tasks
import os

from firecrawl import FirecrawlApp
from deep_research_lib.ai.providers import o3_mini_model, trim_prompt, get_llm_provider # Assuming providers.py is in ai dir
from deep_research_lib.prompt import system_prompt
from deep_research_lib.output_manager import OutputManager
from deep_research_lib.ai.text_splitter import RecursiveCharacterTextSplitter # Ensure text_splitter is correctly placed

output = OutputManager()
llm_provider = get_llm_provider()

def log(*args):
    output.log(*args)

class ResearchProgress(BaseModel):
    currentDepth: int
    totalDepth: int
    currentBreadth: int
    totalBreadth: int
    currentQuery: Optional[str] = None
    totalQueries: int
    completedQueries: int

class ResearchResult(BaseModel):
    learnings: List[str] = Field(default_factory=list) # Initialize as empty list to avoid None issues
    visitedUrls: List[str] = Field(default_factory=list) # Initialize as empty list


CONCURRENCY_LIMIT = 2 # You can make this configurable via env var later

firecrawl_api_key = os.environ.get("FIRECRAWL_KEY") or '' # Default to empty string if not set
if not firecrawl_api_key:
    raise ValueError("FIRECRAWL_KEY environment variable must be set.")
firecrawl = FirecrawlApp(firecrawl_api_key)

class SerpQueriesResponse(BaseModel):
    queries: List[Dict[str, str]] = Field(description="List of SERP queries, max of numQueries")

async def generate_serp_queries(query: str, num_queries: int = 3, learnings: Optional[List[str]] = None) -> List[Dict[str, str]]:
    log(f"generate_serp_queries: Starting query='{query}', num_queries={num_queries}, learnings={learnings}") # LOG
    format_str_serp_queries = """{
      "queries": [
        {"query": "SERP Query 1", "researchGoal": "Research Goal 1"},
        {"query": "SERP Query 2", "researchGoal": "Research Goal 2"},
        ...
      ]
    }""" # Define expected format

    prompt_text = f"""Given the following prompt from the user, generate a list of SERP queries to research the topic.
Return a maximum of {num_queries} queries, but feel free to return less if the original prompt is clear.
Make sure each query is unique and not similar to each other: <prompt>{query}</prompt>\n\n"""
    if learnings:
        prompt_text += f"Here are some learnings from previous research, use them to generate more specific queries: {chr(10).join(learnings)}" # use chr(10) for newline

    # Add format instruction to prompt
    prompt_text += f"\n\nOutput should be in JSON format: {format_str_serp_queries}"

    log(f"generate_serp_queries: Sending prompt to LLM: {prompt_text}") # LOG
    response = await llm_provider.generate_object( # Use llm_provider here
        prompt=prompt_text,
        system=system_prompt(),
        response_model=SerpQueriesResponse # Pydantic model for schema
    )

    log(f"generate_serp_queries: LLM Response received") # LOG
    log(f"generate_serp_queries: Raw LLM Response Object: {response}") # LOG - Log the raw response object
    log(f"generate_serp_queries: Created {len(response.queries)} queries: {response.queries}")
    return response.queries[:num_queries]


class SerpResultResponse(BaseModel):
    learnings: List[str] = Field(description="List of learnings, max of numLearnings")
    followUpQuestions: List[str] = Field(description="List of follow-up questions to research the topic further, max of numFollowUpQuestions")


async def process_serp_result(query: str, result: Any, num_learnings: int = 3, num_follow_up_questions: int = 3) -> SerpResultResponse: # result type as Any for now
    log(f"process_serp_result: Starting query='{query}', num_learnings={num_learnings}, num_follow_up_questions={num_follow_up_questions}") # LOG
    format_str_serp_result = """{
      "learnings": ["Learning 1", "Learning 2", ...],
      "followUpQuestions": ["Question 1", "Question 2", ...]
    }"""
    contents = [item['markdown'] for item in result['data'] if item.get('markdown')] # Safely get markdown content
    contents = [trim_prompt(content, 25000) for content in contents]
    log(f"process_serp_result: Ran query: '{query}', found {len(contents)} contents")

    content_tags = [f'<content>\n{content}\n</content>' for content in contents] # Create content tags separately
    contents_string = chr(10).join(content_tags) # Join with newline

    prompt_text = f"""Given the following contents from a SERP search for the query <query>{query}</query>,
generate a list of learnings from the contents. Return a maximum of {num_learnings} learnings, but feel free to return less if the contents are clear.
Make sure each learning is unique and not similar to each other. The learnings should be concise and to the point, as detailed and information dense as possible.
Make sure to include any entities like people, places, companies, products, things, etc in the learnings, as well as any exact metrics, numbers, or dates.
The learnings will be used to research the topic further.\n\n<contents>{contents_string}</contents>""" # use chr(10) for newline

    prompt_text += f"\n\nOutput should be in JSON format: {format_str_serp_result}"

    log(f"process_serp_result: Sending prompt to LLM: {prompt_text}") # LOG
    response = await llm_provider.generate_object( # Use llm_provider
        prompt=prompt_text,
        system=system_prompt(),
        response_model=SerpResultResponse, # Pydantic model
        # abortSignal=AbortSignal.timeout(60_000), # No AbortSignal in Python asyncio directly, consider timeouts in httpx/openai calls
    )
    log(f"process_serp_result: LLM Response received") # LOG
    log(f"process_serp_result: Created {len(response.learnings)} learnings: {response.learnings}")
    return response


class FinalReportResponse(BaseModel):
    reportMarkdown: str = Field(description="Final report on the topic in Markdown")

async def write_final_report(prompt: str, learnings: List[str], visited_urls: List[str]) -> str:
    log(f"write_final_report: Starting") # LOG
    # format_str_final_report = """{
    #   "reportMarkdown": "Final report in markdown format..."
    # }""" # Define expected format
    learnings_string = trim_prompt(chr(10).join([f"<learning>\n{learning}\n</learning>" for learning in learnings]), 150000) # use chr(10) for newline

    prompt_text = f"""Given the following prompt from the user, write a final report on the topic using the learnings from research.
Make it as as detailed as possible, aim for 3 or more pages, include ALL the learnings from research:\n\n<prompt>{prompt}</prompt>\n\n
Here are all the learnings from previous research:\n\n<learnings>\n{learnings_string}\n</learnings>"""

    # prompt_text += f"\n\nOutput should be in JSON format: {format_str_final_report}"

    log(f"write_final_report: Sending prompt to LLM: {prompt_text}") # LOG
    response = await llm_provider.generate_object( # Use llm_provider
        prompt=prompt_text,
        system=system_prompt(),
        response_model=FinalReportResponse # Pydantic model
    )
    log(f"write_final_report: LLM Response received") # LOG

    urls_section = f"\n\n## Sources\n\n{chr(10).join([f'- {url}' for url in visited_urls])}" # use chr(10) for newline
    return response.reportMarkdown + urls_section


async def deep_research(query: str, breadth: int, depth: int, learnings: Optional[List[str]] = None, visited_urls: Optional[List[str]] = None, on_progress: Optional[Callable[[ResearchProgress, ], None]] = None) -> ResearchResult:
    log(f"deep_research: Starting query='{query}', breadth={breadth}, depth={depth}, learnings={learnings}, visited_urls={visited_urls}") # LOG
    progress = ResearchProgress(
        currentDepth=depth,
        totalDepth=depth,
        currentBreadth=breadth,
        totalBreadth=breadth,
        totalQueries=0,
        completedQueries=0,
    )

    def report_progress(update: Dict[str, Any]): # Simple function to update and report progress
        nonlocal progress # Allow modification of outer scope 'progress'
        for key, value in update.items():
            setattr(progress, key, value) # Dynamically update progress attributes
        if on_progress:
            on_progress(progress) # Call progress callback

    report_progress({"totalQueries": 0, "currentQuery": "Initializing..."}) # Initial progress update

    log(f"deep_research: Calling generate_serp_queries...") # LOG
    serp_queries_data = await generate_serp_queries(query=query, learnings=learnings, num_queries=breadth)
    log(f"deep_research: generate_serp_queries returned, queries_count: {len(serp_queries_data)}") # LOG
    log(f"deep_research: serp_queries_data content BEFORE TASK CREATION: {serp_queries_data}") # LOG - CHECKPOINT: Log serp_queries_data content

    report_progress({"totalQueries": len(serp_queries_data), "currentQuery": serp_queries_data[0]['query'] if serp_queries_data else None})

    limit = asyncio.Semaphore(CONCURRENCY_LIMIT)
    async def process_query_with_limit(serp_query_item): # Define inner function for semaphore
        log(f"process_query_with_limit: ENTERED for query: '{serp_query_item['query']}'") # LOG - Entry point
        async with limit: # Acquire semaphore before processing
            query_text = serp_query_item['query']
            research_goal = serp_query_item['researchGoal']

            log(f"process_query_with_limit: Processing query_text='{query_text}', research_goal='{research_goal}'") # LOG

            try:
                log(f"process_query_with_limit: Calling firecrawl.search for query: '{query_text}'") # LOG - Before firecrawl.search
                # result = await firecrawl.search(query_text, options={"timeout": 15000, "limit": 5, "scrapeOptions": {"formats": ["markdown"]}}) # Example options
                result = firecrawl.search(query_text) # Example options
                log(f"process_query_with_limit: firecrawl.search result received for query: '{query_text}'") # LOG - After firecrawl.search
                log(f"process_query_with_limit: firecrawl.search result CONTENT: {result}") # LOG - CHECKPOINT: Log firecrawl result CONTENT

                new_urls = [item['url'] for item in result['data'] if item.get('url')] # Safely extract URLs
                new_breadth = max(1, breadth // 2) # Ensure new_breadth is at least 1
                new_depth = depth - 1

                # *** COMMENT OUT process_serp_result COMPLETELY - FOR TESTING FIRECRAWL ***
                # log(f"process_query_with_limit: Calling process_serp_result for query: '{query_text}'") # LOG - Before process_serp_result
                # serp_result_data = await process_serp_result(
                #     query=query_text, result=result, num_follow_up_questions=new_breadth
                # )
                # log(f"process_query_with_limit: process_serp_result response received for query: '{query_text}'") # LOG - After process_serp_result


                all_learnings = (learnings or []) #+ serp_result_data.learnings # Commented out: No learnings from process_serp_result
                # Modified to use Firecrawl data directly as "learnings" for testing:
                all_learnings = (learnings or []) + [item['markdown'] for item in result['data'] if item.get('markdown')] # Use markdown content as learnings for test

                all_urls = (visited_urls or []) + new_urls # Use empty list if visited_urls is None

                return ResearchResult(learnings=all_learnings, visitedUrls=all_urls) # Return ResearchResult object with learnings (now from firecrawl)


            except Exception as e:
                error_message = str(e)
                log(f"process_query_with_limit: Exception caught for query: '{query_text}'. Error: {e}") # LOG
                if "Timeout" in error_message:
                    log(f"process_query_with_limit: Timeout error running query: {query_text}: {e}")
                else:
                    log(f"process_query_with_limit: Error running query: {query_text}: {e}")
                return ResearchResult(learnings=["MOCK LEARNING - ERROR Query (Firecrawl call issue): " + query_text + " Error: " + error_message], visitedUrls=[]) # Modified return

    tasks = [process_query_with_limit(serp_query) for serp_query in serp_queries_data] # Create tasks for concurrent processing
    log(f"deep_research: Created {len(tasks)} tasks for process_query_with_limit") # LOG
    results = await asyncio.gather(*tasks) # Gather results from all tasks
    log(f"deep_research: All tasks completed, gathered results") # LOG
    log(f"deep_research: Results from tasks: {results}") # LOG - Log the 'results' variable!


    # Combine and deduplicate learnings and visited URLs from all results
    all_learnings_set: Set[str] = set()
    all_visited_urls_set: Set[str] = set()

    for res in results:
        if res and res.learnings: # Check if res is not None and has learnings
            all_learnings_set.update(res.learnings)
            all_visited_urls_set.update(res.visitedUrls)

    flat_learnings = list(all_learnings_set) # Convert set back to list
    log(f"deep_research: Returning ResearchResult with {len(flat_learnings)} learnings and {len(all_visited_urls_set)} visited URLs") # LOG

    return ResearchResult(learnings=flat_learnings, visitedUrls=list(all_visited_urls_set))