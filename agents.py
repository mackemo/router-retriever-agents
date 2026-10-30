import os
import json
import requests
from dotenv import load_dotenv
from pathlib import Path
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from crewai_tools import PDFSearchTool


# load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# grab dataset
PDF_PATH = Path(__file__).parent / "data" / "transformer_research_paper.pdf"


# create the model
# ------------------------
MODEL_NAME = "gpt-4o-mini"

llm = LLM(
    model=f"openai/{MODEL_NAME}",
    temperature=0,
    api_key=OPENAI_API_KEY,
)


# ---- Custom CrewAI Tool for Web Search ----
class TavilySearchTool(BaseTool):
    name: str = "Web Search"
    description: str = "Search the web for recent information."

    def _run(self, query: str):
        url = "https://api.tavily.com/search"

        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": 3
        }

        response = requests.post(url, json=payload)
        
        data = response.json()

        results = []
        for r in data["results"]:
            results.append(f"{r['title']} - {r['url']}")

        return "\n".join(results)


web_search_tool = TavilySearchTool()
pdf_search_tool = PDFSearchTool(pdf=str(PDF_PATH))
# ------------------------


# Agents
# ------------------------
# Router agent
router = Agent(
    role="Router",
    goal="Choose PDF for questions about the supplied paper, otherwise choose WEB.",
    backstory="You route each question to the best information source.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Retriever agent
retriever = Agent(
    role="Retriever",
    goal="Use the selected search tool to answer the user's question.",
    backstory="You answer using retrieved evidence and identify the source.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[web_search_tool, pdf_search_tool]
)


# -----------------------
# User Question
# ------------------------
question = input("Enter your question: ").strip()


# Tasks
# ------------------------
task_route= Task(
    description=f"For this question, return exactly PDF if it is about the supplied Transformer paper; otherwise return WEB. Question: {question}",
    expected_output="PDF or WEB",
    agent=router
)


task_retrieve = Task(
    description=f"""
Based on the router's output for which information source to pull from, pick the tool that corresponds to the source and use it to answer the user's question.

Requirements:
- Choose the pdf_search_tool if the router's output is PDF, otherwise choose the web_search_tool.
""",
    expected_output="A concise answer grounded in retrieved evidence.",
    agent=retriever,
    context=[task_route]
)


# ---- Crew ----
crew = Crew(
    agents=[router, retriever],
    tasks=[task_route, task_retrieve],
    verbose=True
)

result = crew.kickoff()

print("\nFinal Output:\n")
print(result.raw)