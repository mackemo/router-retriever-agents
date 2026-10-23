import os
import json
from dotenv import load_dotenv
from pathlib import Path
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
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