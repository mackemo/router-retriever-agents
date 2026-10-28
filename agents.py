import os
import json
import requests
from dotenv import load_dotenv
from pathlib import Path
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from crewai_tools import PDFSearchTool
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI


# load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# grab dataset
PDF_PATH = "data/transformer_research_paper.pdf"


# create the model
MODEL_NAME = "gpt-3.5-turbo"

llm = ChatOpenAI(
    model_name=MODEL_NAME,
    temperature=0,
    openai_api_key=[OPENAI_API_KEY]
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


search_tool = TavilySearchTool()



# LLM
# ------------------------------
# initialize the llm
MODEL_NAME = "gpt-3.5-turbo"

llm = ChatOpenAI(
    model_name=MODEL_NAME,
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)