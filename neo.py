import sqlite3
import spacy
import time
from neo4j import GraphDatabase
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.llms import LlamaCpp
import streamlit as st
from PIL import Image
from chromadb.utils import embedding_functions
import chromadb
import json

# Decay and growth rates for memory evolution
DECAY_RATE = 0.05
GROWTH_RATE = 0.1
# Initialize and manage SQLite database
def setup_sqlite_db(path):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    
    # Table for storing nodes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS node_data (
            node_id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            activation_rate FLOAT DEFAULT 1.0,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Table for edges with strength between nodes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS edge_data (
            edge_id TEXT PRIMARY KEY,
            source_node_id TEXT NOT NULL,
            target_node_id TEXT NOT NULL,
            strength FLOAT DEFAULT 1.0,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_node_id) REFERENCES node_data(node_id),
            FOREIGN KEY (target_node_id) REFERENCES node_data(node_id)
        );
    """)
    
    # History table for node activation rates
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS node_activation_history (
            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL,
            previous_activation_rate FLOAT,
            new_activation_rate FLOAT,
            update_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (node_id) REFERENCES node_data(node_id)
        );
    """)
    
    # History table for edge strengths
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS edge_strength_history (
            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            edge_id TEXT NOT NULL,
            previous_strength FLOAT,
            new_strength FLOAT,
            update_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (edge_id) REFERENCES edge_data(edge_id)
        );
    """)

    # Table for storing conversation history data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history_data (
            conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation TEXT NOT NULL,
            updatetime DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    print("SQLite tables created.")
    conn.commit()
    conn.close()
# Load spaCy model and enhance symbolic representation
class SymbolicRepresentation:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print(f"Failed to load spaCy model: {e}")
            self.nlp = None

    def text_to_symbol(self, text):
        if not self.nlp:
            return []
        doc = self.nlp(text)
        symbols = []
        for token in doc:
            basic_symbol = f"{token.lemma_}:{token.pos_}"
            symbols.append(basic_symbol)
            if token.dep_ != "punct":
                relation = f"{token.dep_}({token.head.lemma_}, {token.lemma_})"
                symbols.append(relation)
            if token.head.pos_ == "VERB":
                action = f"action({token.head.lemma_}, {token.lemma_})"
                symbols.append(action)
            for child in token.children:
                if child.dep_ != "punct":
                    relation = f"{child.dep_}({token.lemma_}, {child.lemma_})"
                    symbols.append(relation)
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        symbols.extend(noun_chunks)
        symbols.extend([f"entity({ent[0]}:{ent[1]})" for ent in entities])
        return symbols

# Neo4j graph database integration for memory management
class GraphMemoryLayer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.neo4j_driver.close()

    def update_graph(self, user_input, ai_response, user_feedback=None):
        current_time = time.time()
        feedback_weight = 0.2 if user_feedback == "positive" else -0.1 if user_feedback == "negative" else 0
        with self.driver.session() as session:
            session.execute_write(self._create_and_link, user_input, ai_response, current_time, feedback_weight)
    
    def reset_memory(self):
        with self.driver.session() as session:
            session.write_transaction(self._clear_graph)

    @staticmethod
    def _clear_graph(tx):
        query = "MATCH (n) DETACH DELETE n"
        tx.run(query)
        print("Graph memory has been reset.")

    @staticmethod
    def _create_and_link(tx, user_input, ai_response, timestamp, feedback_weight):
        query = """
        MERGE (u:UserInput {text: $user_input})
        ON CREATE SET u.activation_rate = 1.0, u.created_at = $timestamp
        ON MATCH SET u.activation_rate = u.activation_rate * (1 - $decay_rate) + $growth_rate,
                    u.last_updated = $timestamp
        MERGE (a:AIResponse {text: $ai_response})
        ON CREATE SET a.activation_rate = 1.0, a.created_at = $timestamp
        ON MATCH SET a.activation_rate = a.activation_rate * (1 - $decay_rate) + $growth_rate,
                    a.last_updated = $timestamp
        MERGE (u)-[r:RESPONDS_TO]->(a)
        ON CREATE SET r.strength = 1.0 + $feedback_weight, r.created_at = $timestamp
        ON MATCH SET r.strength = r.strength * (1 - $decay_rate) + $growth_rate + $feedback_weight,
                    r.last_updated = $timestamp
        """
        tx.run(query, user_input=user_input, ai_response=ai_response, timestamp=timestamp, decay_rate=DECAY_RATE, growth_rate=GROWTH_RATE, feedback_weight=feedback_weight)


    def query_context(self, query_terms):
        with self.driver.session() as session:
            return session.execute_read(self._find_relevant_texts, query_terms)


    @staticmethod
    def _find_relevant_texts(tx, query_terms):
        query = """
        MATCH (n)
        WHERE any(term in $query_terms WHERE term IN n.text)
        RETURN n.text as text, n.timestamp as timestamp
        ORDER BY n.timestamp DESC
        """
        result = tx.run(query, query_terms=query_terms)
        return [(record["text"], record["timestamp"]) for record in result]

    def recall_context(self):
        with self.driver.session() as session:
            return session.execute_read(self._recall_recent_texts)

    @staticmethod
    def _recall_recent_texts(tx):
        query = """
        MATCH (n:UserInput)
        RETURN n.text as text
        ORDER BY n.timestamp DESC
        """
        result = tx.run(query)
        return ' '.join([record["text"] for record in result])
    
        ################################## revised code ##################################
    
    def get_all_nodes(self):
        with self.driver.session() as session:
            return session.read_transaction(self._get_all_nodes)

    @staticmethod
    def _get_all_nodes(tx):
        query = """
        MATCH (n)
        RETURN labels(n) as labels, properties(n) as properties
        """
        result = tx.run(query)
        return [{'labels': record['labels'], 'properties': record['properties']} for record in result]

    def get_all_edges(self):
        with self.driver.session() as session:
            return session.read_transaction(self._get_all_edges)

    @staticmethod
    def _get_all_edges(tx):
        query = """
        MATCH ()-[r]->()
        RETURN type(r) as type, properties(r) as properties
        """
        result = tx.run(query)
        return [{'type': record['type'], 'properties': record['properties']} for record in result]
    
    ################################## revised code ##################################

class ChatAI:
    def __init__(self, uri, user, password, sqlite_db_path):
        self.symbolic_rep = SymbolicRepresentation()
        self.embedding_model = embedding_functions.DefaultEmbeddingFunction()
        self.client = chromadb.PersistentClient(path="./chroma_client")
        self.collection = self.client.get_or_create_collection(name="collection_chat", embedding_function=self.embedding_model)
        self.memory_layer = GraphMemoryLayer(uri, user, password)
        self.neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
        self.sqlite_db_path = sqlite_path


    def model(self, question, multi_response=True):
        try:
            llm = LlamaCpp(model_path="./mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                        n_ctx=32768, n_threads=8, n_gpu_layers=-1, device='gpu',
                        temperature=1.2, top_p=0.9)  # Adjusted for more diversity
            conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory())

            context = self.memory_layer.recall_context() + ' ' + question
            if multi_response:
                responses = [conversation.predict(input=f"{context} Option {i}").split("\n")[0] for i in range(1, 4)]
                selected_response = self.evaluate_responses(question, responses)
                response = selected_response
            else:
                response = conversation.predict(input=context).split("\n")[0]

            self.memory_layer.update_graph(question, response)

            return response
        except Exception as e:
            print(f"Error during model prediction: {e}")
            return "Sorry, I encountered an error processing your request."

    def evaluate_responses(self, question, responses):
        question_symbols = set(self.symbolic_rep.text_to_symbol(question))
        response_symbols = [set(self.symbolic_rep.text_to_symbol(response)) for response in responses]
        recent_context_symbols = set()
        for text, _ in self.memory_layer.query_context([question]):
            recent_context_symbols.update(self.symbolic_rep.text_to_symbol(text))
        
        scores = []
        for response_set in response_symbols:
            relevance_to_question = len(response_set & question_symbols)
            relevance_to_context = len(response_set & recent_context_symbols)
            score = relevance_to_question + relevance_to_context
            scores.append(score)
        
        best_response_index = scores.index(max(scores))
        return responses[best_response_index]

    def insert_sqllite3(self, conversation):
        with sqlite3.connect(self.sqlite_db_path) as conn:
            try:
                cur = conn.cursor()
                insert_chat_sql = """INSERT INTO chat_history_data (conversation) VALUES (?)"""
                cur.execute(insert_chat_sql, (json.dumps(conversation),))
                conn.commit()  # Ensure changes are committed
                print("Data inserted successfully")
            except sqlite3.Error as e:
                print(f"SQLite error: {e}")


    def get_data_from_sqllite3(self):
        with sqlite3.connect(self.sqlite_db_path) as conn:
            cur = conn.cursor()
            res = cur.execute("SELECT * FROM chat_history_data")
            result_data = [{"conversation_id": row[0], "conversation": json.loads(row[1]), "updatetime": row[2]} for row in res.fetchall()]
            if not result_data:
                print("No data found.")
            return result_data


    def chatbot_ui(self):
        st.title("Chat Bot")
        if "conversation" not in st.session_state:
            st.session_state["conversation"] = []

        feedback = None
        for msg in st.session_state["conversation"]:
            if msg["role"] == "user":
                st.chat_message("user").markdown(msg["content"])
            else:
                feedback = st.radio("How relevant was this response?", ("Positive", "Negative", "Neutral"), key=msg["content"])
                st.chat_message("assistant").markdown(msg["content"])
                if feedback:
                    self.memory_layer.update_graph(msg["content"], feedback, user_feedback=feedback.lower())

        if questions := st.chat_input():
            response = self.model(questions)
            st.session_state["conversation"].append({"role": "user", "content": questions})
            st.session_state["conversation"].append({"role": "assistant", "content": response})
            st.rerun()

        if st.button("End Conversation"):
            self.insert_sqllite3(st.session_state["conversation"])

            ############################## revised code ##############################
            nodes = self.memory_layer.get_all_nodes()
            self.sync_nodes_to_sqlite(nodes)

            edges = self.memory_layer.get_all_edges()
            self.sync_edges_to_sqlite(edges)
            ############################## revised code ##############################
            st.session_state["conversation"] = []
            st.rerun()

        if st.button("Start New Conversation"):
            if st.session_state["conversation"]:
                self.insert_sqllite3(st.session_state["conversation"])
            st.session_state["conversation"] = []
            st.rerun()

    def menu(self):
        menu = ['Home', 'Chat Bot', 'Chat History']
        choice = st.sidebar.radio('Menu', menu)
        if choice == 'Home':
            st.title("Welcome to Chat Bot!")
            image_path = "./chat_bot.png"
            try:
                image = Image.open(image_path)
                st.image(image=image, use_column_width=True)
            except FileNotFoundError:
                st.error("Image not found. Please check the file path.")
        elif choice == 'Chat Bot':
            self.chatbot_ui()
        elif choice == 'Chat History':
            self.view_chat_history()


    def view_chat_history(self):
        st.title("Chat History")
        res = self.get_data_from_sqllite3()
        for idx, item in enumerate(res):
            conversation_id = item['conversation_id']
            conversation = item['conversation']
            updatetime = item['updatetime']
            conversation_text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in conversation)
            st.text_area(label=f"Conversation {conversation_id} from {updatetime}", value=conversation_text, height=200)


    def restart_model(self):
        self.symbolic_rep = SymbolicRepresentation()
        self.embedding_model = embedding_functions.DefaultEmbeddingFunction()
        self.client = chromadb.PersistentClient(path="./chroma_client")
        self.collection = self.client.get_or_create_collection(name="collection_chat", embedding_function=self.embedding_model)
        self.memory_layer = GraphMemoryLayer(uri, user, password)


    # def sync_node_to_sqlite(node_id, activation_rate, db_path='./chat_ai.db'):
    #     with sqlite3.connect(db_path) as conn:
    #         cursor = conn.cursor()
    #         cursor.execute("""
    #             INSERT INTO node_data (node_id, activation_rate) VALUES (?, ?)
    #             ON CONFLICT(node_id) DO UPDATE SET activation_rate=excluded.activation_rate, last_updated=CURRENT_TIMESTAMP
    #         """, (node_id, activation_rate))
    #         conn.commit()


    ################################## revised code ################################## 下面啊
    def sync_nodes_to_sqlite(self, nodes, db_path='./chat_ai.db'):
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            for node in nodes:
                node_id = node['properties'].get('node_id')
                activation_rate = node['properties'].get('activation_rate')
                if node_id and activation_rate:
                    cursor.execute("""
                        INSERT INTO node_data (node_id, activation_rate) VALUES (?, ?)
                        ON CONFLICT(node_id) DO UPDATE SET activation_rate=excluded.activation_rate, last_updated=CURRENT_TIMESTAMP
                    """, (node_id, activation_rate))
            conn.commit()

    ################################## revised code ##################################
    
    # def sync_edge_to_sqlite(edge_id, source_node_id, target_node_id, strength, db_path='./chat_ai.db'):
    #     with sqlite3.connect(db_path) as conn:
    #         cursor = conn.cursor()
    #         cursor.execute("""
    #             INSERT INTO edge_data (edge_id, source_node_id, target_node_id, strength) VALUES (?, ?, ?, ?)
    #             ON CONFLICT(edge_id) DO UPDATE SET strength=excluded.strength, last_updated=CURRENT_TIMESTAMP
    #         """, (edge_id, source_node_id, target_node_id, strength))
    #         conn.commit()

        ################################## revised code ##################################下面！
    def sync_edges_to_sqlite(self, edges, db_path='./chat_ai.db'):
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            for edge in edges:
                edge_id = edge['properties'].get('edge_id')
                activation_rate = edge['properties'].get('activation_rate')
                if edge_id and activation_rate:
                    cursor.execute("""
                        INSERT INTO edge_data (edge_id, activation_rate) VALUES (?, ?)
                        ON CONFLICT(edge_id) DO UPDATE SET activation_rate=excluded.activation_rate, last_updated=CURRENT_TIMESTAMP
                    """, (edge_id, activation_rate))
            conn.commit()
        
        ################################## revised code ##################################
def update_node_activation_rate(node_id, new_rate):
    with sqlite3.connect('./chat_ai.db') as conn:
        cur = conn.cursor()
        # Fetch the current rate
        cur.execute("SELECT activation_rate FROM node_data WHERE node_id = ?", (node_id,))
        current_rate = cur.fetchone()[0]
        
        # Update the node table
        cur.execute("UPDATE node_data SET activation_rate = ? WHERE node_id = ?", (new_rate, node_id))
        
        # Log the change
        cur.execute("""
            INSERT INTO node_activation_history (node_id, previous_activation_rate, new_activation_rate)
            VALUES (?, ?, ?)
        """, (node_id, current_rate, new_rate))
        conn.commit()

def update_edge_strength(edge_id, new_strength):
    with sqlite3.connect('./chat_ai.db') as conn:
        cur = conn.cursor()
        # Fetch the current strength
        cur.execute("SELECT strength FROM edge_data WHERE edge_id = ?", (edge_id,))
        current_strength = cur.fetchone()[0]
        
        # Update the edge table
        cur.execute("UPDATE edge_data SET strength = ? WHERE edge_id = ?", (new_strength, edge_id))
        
        # Log the change
        cur.execute("""
            INSERT INTO edge_strength_history (edge_id, previous_strength, new_strength)
            VALUES (?, ?, ?)
        """, (edge_id, current_strength, new_strength))
        conn.commit()

def setup_neo4j_db(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:RELATES) REQUIRE r.id IS UNIQUE")
            print("Neo4j constraints setup.")
    except Exception as e:
        print(f"Error setting up Neo4j constraints: {e}")
    finally:
        driver.close()


# Main function to coordinate setup and export

sqlite_path = './chat_ai.db'
neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "52900000"
    
# Assuming the Neo4j driver setup is already configured globally or passed around appropriately
uri = "bolt://localhost:7687"
user = "neo4j"
password = "52900000"
setup_sqlite_db(sqlite_path)
setup_neo4j_db(neo4j_uri, neo4j_user, neo4j_password)
chat = ChatAI(neo4j_uri, neo4j_user, neo4j_password, sqlite_path)

neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
chat.menu()
