import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BraveSearch

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

llm = ChatGroq(temperature=0, model_name="llama2-70b-4096", groq_api_key=GROQ_API_KEY)

search = BraveSearch.from_api_key(api_key=BRAVE_API_KEY, search_kwargs={"count": 3})

tools = [
    Tool(
        name="Brave Search",
        func=search.run,
        description="Useful for searching the web for current information."
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def main():
    query = "What are the latest developments in AI?"
    result = agent.run(query)
    print(result)

if __name__ == "__main__":
    main()
