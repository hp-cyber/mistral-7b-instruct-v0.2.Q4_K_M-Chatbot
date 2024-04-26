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

    def extract_intent(self, text):
        doc = self.nlp(text)
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                return f"intent({token.lemma_})"
        return "intent(unknown)"

# Neo4j graph database integration for memory management
class GraphMemoryLayer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def update_graph(self, user_input, ai_response, user_feedback=None):
        current_time = time.time()
        feedback_weight = 0.2 if user_feedback == "positive" else -0.1 if user_feedback == "negative" else 0
        with self.driver.session() as session:
            session.write_transaction(self._create_and_link, user_input, ai_response, current_time, feedback_weight)

    @staticmethod
    def _create_and_link(tx, user_input, ai_response, timestamp, feedback_weight):
        query = """
        MERGE (u:UserInput {text: $user_input})
        ON CREATE SET u.activation_rate = 1.0, u.created_at = $timestamp
        ON MATCH SET u.activation_rate = u.activation_rate * exp(-$decay_rate * ($timestamp - u.last_updated)) + $growth_rate,
                    u.last_updated = $timestamp
        MERGE (a:AIResponse {text: $ai_response})
        ON CREATE SET a.activation_rate = 1.0, a.created_at = $timestamp
        ON MATCH SET a.activation_rate = a.activation_rate * exp(-$decay_rate * ($timestamp - a.last_updated)) + $growth_rate,
                    a.last_updated = $timestamp
        MERGE (u)-[r:RESPONDS_TO]->(a)
        ON CREATE SET r.strength = 1.0, r.created_at = $timestamp
        ON MATCH SET r.strength = r.strength * exp(-$decay_rate * ($timestamp - r.last_updated)) + $feedback_weight,
                    r.last_updated = $timestamp
        """
        tx.run(query, user_input=user_input, ai_response=ai_response, timestamp=timestamp, decay_rate=DECAY_RATE, growth_rate=GROWTH_RATE, feedback_weight=feedback_weight)

    def query_context(self, query_terms):
        with self.driver.session() as session:
            return session.read_transaction(self._find_relevant_texts, query_terms)

    @staticmethod
    def _find_relevant_texts(tx, query_terms):
        query = """
        MATCH (n)
        WHERE any(term in $query_terms WHERE term IN n.text)
        RETURN n.text as text, n.timestamp as timestamp
        ORDER BY n.timestamp DESC LIMIT 3
        """
        result = tx.run(query, query_terms=query_terms)
        return [(record["text"], record["timestamp"]) for record in result]

    def recall_context(self):
        with self.driver.session() as session:
            return session.read_transaction(self._recall_recent_texts)

    @staticmethod
    def _recall_recent_texts(tx):
        query = """
        MATCH (n:UserInput)
        RETURN n.text as text
        ORDER BY n.timestamp DESC LIMIT 3
        """
        result = tx.run(query)
        return ' '.join([record["text"] for record in result])

class ChatAI:
    def __init__(self, uri, user, password):
        self.symbolic_rep = SymbolicRepresentation()
        self.embedding_model = embedding_functions.DefaultEmbeddingFunction()
        self.client = chromadb.PersistentClient(path="./chroma_client")
        self.collection = self.client.get_or_create_collection(name="collection_chat", embedding_function=self.embedding_model)
        self.memory_layer = GraphMemoryLayer(uri, user, password)

    def model(self, question, multi_response=False):
        try:
            llm = LlamaCpp(model_path="./mistral-7b-instruct-v0.2.Q4_K_M.gguf", n_ctx=32768, n_threads=8, n_gpu_layers=-1)
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

    def create_table(self):
        with sqlite3.connect('./chat_ai.db') as conn:
            try:
                cur = conn.cursor()
                sql = """CREATE TABLE IF NOT EXISTS chat_history_data (
                                            conversation_id INTEGER PRIMARY KEY,
                                            conversation TEXT, 
                                            updatetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                        );"""
                cur.execute(sql)
            except sqlite3.Error as e:
                print(f"SQLite error: {e}")

    def insert_sqllite3(self, conversation):
        with sqlite3.connect('./chat_ai.db') as conn:
            try:
                cur = conn.cursor()
                insert_sql = """INSERT INTO chat_history_data (conversation) VALUES (?)"""
                cur.execute(insert_sql, (json.dumps(conversation),))
            except sqlite3.Error as e:
                print(f"SQLite error: {e}")

    def get_data_from_sqllite3(self):
        try:
            with sqlite3.connect('./chat_ai.db') as conn:
                cur = conn.cursor()
                res = cur.execute("SELECT * FROM chat_history_data")
                result_data = [{"conversation_id": row[0], "conversation": json.loads(row[1]), "updatetime": row[2]} for row in res.fetchall()]
            return result_data
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return []

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
            st.session_state["conversation"] = []
            st.rerun()

        if st.button("Start New Conversation"):
            if st.session_state["conversation"]:
                self.insert_sqllite3(st.session_state["conversation"])
            st.session_state["conversation"] = []
            st.rerun()

    def menu(self):
        self.create_table()
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

uri = "bolt://localhost:7687"
user = "neo4j"
password = "52900000"  # Change to your actual password
chat = ChatAI(uri, user, password)
chat.menu()



