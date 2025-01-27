# main.py
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from langgraph.graph import StateGraph, END

# Database connections
class DBMigrator:
    def __init__(self):
        self.source_conn = None
        self.target_conn = None

    def configure_postgres(self, **creds):
        import psycopg2
        self.source_conn = psycopg2.connect(**creds)
    
    def configure_mysql(self, **creds):
        import mysql.connector
        self.target_conn = mysql.connector.connect(**creds)

# Query generator using LangChain
class QueryTranslator:
    def __init__(self):
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
        
        self.llm = ChatOpenAI(model="gpt-4")
        self.prompt = ChatPromptTemplate.from_template(
            """Convert PostgreSQL to MySQL query considering syntax, functions, and data types:
            PostgreSQL: {query}
            MySQL Equivalent:"""
        )

    def translate(self, query):
        return self.prompt | self.llm | (lambda x: x.content)

# Workflow visualization

if 'workflow_steps' not in st.session_state:
    st.session_state.workflow_steps = {
        "config": "pending",
        "listen": "pending",
        "generate": "pending"
    }


def draw_workflow():
    status_colors = {
        "pending": "skyblue",
        "active": "#FFA500",  # Orange
        "completed": "#90EE90",  # Light Green
        "error": "#FF474C"
    }
    
    G = nx.DiGraph()
    G.add_nodes_from([
        ("Config", {"status": st.session_state.workflow_steps["config"]}),
        ("Listen", {"status": st.session_state.workflow_steps["listen"]}),
        ("Generate", {"status": st.session_state.workflow_steps["generate"]})
    ])
    
    G.add_edges_from([("Config", "Listen"), ("Listen", "Generate")])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    pos = nx.spring_layout(G, seed=42)
    
    node_colors = [status_colors[G.nodes[n]["status"]] for n in G.nodes]
    labels = {n: f"{n}\n({G.nodes[n]['status']})" for n in G.nodes}
    
    nx.draw(G, pos, with_labels=True, labels=labels, 
            node_color=node_colors, node_size=2500, 
            font_size=10, ax=ax, edge_color="gray")
    return fig

def run_conversion_workflow(user_query):
    try:
        # Reset statuses
        st.session_state.workflow_steps = {k: "pending" for k in st.session_state.workflow_steps}
        
        # Create workflow
        workflow = StateGraph(dict)
        
        # Config Node
        def config_step(state):
            st.session_state.workflow_steps["config"] = "active"
            if not st.session_state.migrator.source_conn or not st.session_state.migrator.target_conn:
                raise ConnectionError("Database connections failed")
            st.session_state.workflow_steps["config"] = "completed"
            return state
        
        # Listen Node
        def listen_step(state):
            st.session_state.workflow_steps["listen"] = "active"
            if not user_query.strip():
                raise ValueError("Empty query received")
            st.session_state.workflow_steps["listen"] = "completed"
            return {"query": user_query}
        
        # Generate Node
        def generate_step(state):
            st.session_state.workflow_steps["generate"] = "active"
            translator = QueryTranslator()
            chain = translator.translate(state["query"])
            result = chain.invoke({"query": state["query"]})
            st.session_state.workflow_steps["generate"] = "completed"
            return {"converted_query": result}
        
        # Build workflow
        workflow.add_node("config", config_step)
        workflow.add_node("listen", listen_step)
        workflow.add_node("generate", generate_step)
        
        workflow.add_edge("config", "listen")
        workflow.add_edge("listen", "generate")
        workflow.add_edge("generate", END)
        
        # Execute workflow
        workflow.set_entry_point("config")
        app = workflow.compile()
        return app.invoke({})
        
    except Exception as e:
        current_step = next(k for k, v in st.session_state.workflow_steps.items() if v == "active")
        st.session_state.workflow_steps[current_step] = "error"
        raise e
    
# Streamlit UI
st.title("AI-Powered Database Migration")

# Configuration
with st.sidebar:
    st.header("Database Credentials")
    
    pg_host = st.text_input("PostgreSQL Host")
    pg_user = st.text_input("PostgreSQL User")
    pg_pass = st.text_input("PostgreSQL Password", type="password")
    
    mysql_host = st.text_input("MySQL Host")
    mysql_user = st.text_input("MySQL User")
    mysql_pass = st.text_input("MySQL Password", type="password")
    
    if st.button("Initialize Connections"):
        migrator = DBMigrator()
        migrator.configure_postgres(
            host=pg_host, user=pg_user, password=pg_pass,
            database="postgres"  # Add additional fields as needed
        )
        migrator.configure_mysql(
            host=mysql_host, user=mysql_user, password=mysql_pass,
            database="mysql"
        )
        st.session_state.migrator = migrator

# Main interface
col1, col2 = st.columns([3, 2])

with col1:
    st.header("Migration Workflow")
    st.pyplot(draw_workflow())

with col2:
    st.header("Query Conversion")
    user_query = st.text_area("Ask AI to migrate")
    
    if st.button("Convert to MySQL"):
        if 'migrator' not in st.session_state:
            st.error("Configure databases first!")
            st.stop()
            
        try:
            with st.spinner("Converting query..."):
                result = run_conversion_workflow(user_query)
                st.subheader("Converted Query")
                st.code(result["converted_query"], language="sql")
                st.success("Conversion completed successfully!")
                
                # Force refresh of workflow visualization
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"Conversion failed: {str(e)}")
            st.experimental_rerun()