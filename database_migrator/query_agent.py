# Query Generation Agent (query_agent.py)
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI

class QueryGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o")
        
    def generate_migration_query(self, user_query):
        prompt = ChatPromptTemplate.from_template(
            """Convert PostgreSQL to MySQL query:
            Original: {query}
            Consider:
            1. Data type differences
            2. Syntax variations
            3. Function equivalents
            """
        )
        chain = prompt | self.llm
        return chain.invoke({"query": user_query})