# Router–Retriever Agents

A small command-line question-answering system built with CrewAI. It routes each question to either a supplied research-paper PDF or a web search tool, then uses a retrieval agent to produce an answer.

## System architecture

The application is implemented in `agents.py` and has four main parts:

- **User input:** The program prompts for one question when it starts.
- **LLM:** Both CrewAI agents use OpenAI's `gpt-4o-mini` model with temperature set to `0`.
- **Knowledge-source tools:** `PDFSearchTool` searches `data/transformer_research_paper.pdf`; a custom CrewAI tool sends queries to the Tavily Search API.
- **CrewAI workflow:** A sequential crew runs the routing task first and the retrieval task second.

## Agents and responsibilities

### Router

The Router decides which source is appropriate. Questions about the supplied Transformer paper should be labeled `PDF`; all other questions should be labeled `WEB`. Its task requests one of those labels so the next task can use the result.

### Retriever

The Retriever receives the router task as context and has access to both search tools. It is instructed to use the PDF tool for a `PDF` route and the Tavily tool otherwise. It returns a concise answer grounded in the selected source.

### Tools

- **PDF search:** Uses CrewAI's `PDFSearchTool` on the local research paper.
- **Web search:** Posts the query to Tavily's search endpoint and returns up to three result titles and URLs. The tool currently returns titles and links, not full page content or search-result excerpts, so the answer may have limited evidence for web questions.

## Coordination flow

1. Run `python3 agents.py` and enter a question at the prompt.
2. The Router task includes the question and produces `PDF` or `WEB`.
3. CrewAI passes the Router task output to the Retriever task through its `context` field.
4. The Retriever selects the matching tool, searches, and drafts an answer.
5. The program prints the final answer.

Both agents are configured with `allow_delegation=False`. Coordination happens through the ordered tasks and task context rather than agent-to-agent delegation.

## Setup and run

Use Python 3.10 or later. Install the packages used by the script:

```bash
pip install crewai crewai-tools python-dotenv requests
```

Create a `.env` file in the project directory with API keys:

```dotenv
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Then start the program:

```bash
python3 agents.py
```

The PDF path is resolved relative to `agents.py`, so the expected file is `data/transformer_research_paper.pdf`.

## Challenges and trade-offs

- **Routing adds an LLM call.** A separate Router makes source selection explicit and easy to inspect, but adds latency and model cost compared with routing through a simple keyword rule or doing both steps in one agent.
- **One retriever has both tools.** Passing the route through task context keeps the workflow small. The Retriever still makes the final tool choice, so it depends on correctly following the Router's label.
- **Web search is lightweight.** Returning only titles and URLs keeps the custom Tavily tool simple, but gives the answerer less source text than returning snippets or fetching pages. Search quality can therefore vary by question.
- **The PDF is local and fixed.** A predictable local path avoids a separate document-upload service, but changing the paper requires replacing the configured file or updating `PDF_PATH`.
- **Secrets and external services are required.** OpenAI and Tavily keys must be available in `.env`, and web questions depend on network access and a valid Tavily response. The script currently does not add retries or structured handling for API errors.
- **The input is one question per run.** This keeps the example easy to use from a terminal; a loop or UI would be needed for a longer interactive session.
