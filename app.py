import sqlite3
import spacy
import time
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.llms import LlamaCpp
import networkx as nx
import streamlit as st
from PIL import Image
from chromadb.utils import embedding_functions
import chromadb
import json

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

        return symbols

    def extract_intent(self, text):
        doc = self.nlp(text)
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                return f"intent({token.lemma_})"
        return "intent(unknown)"

class GraphMemoryLayer:
    def __init__(self):
        self.graph = nx.DiGraph()

    def update_graph(self, user_input, ai_response, user_feedback=None):
        current_time = time.time()
        user_node = f"user_{len(self.graph.nodes) + 1}"
        ai_node = f"ai_{len(self.graph.nodes) + 2}"

        initial_weight = 1 + (0.2 if user_feedback == "positive" else -0.1 if user_feedback == "negative" else 0)
        self.graph.add_node(user_node, type='user', text=user_input, timestamp=current_time, weight=initial_weight)
        self.graph.add_node(ai_node, type='ai', text=ai_response, timestamp=current_time, weight=initial_weight)

        for symbol in set(user_input) & set(ai_response):
            self.graph.add_edge(user_node, ai_node, symbol=symbol, timestamp=current_time, weight=initial_weight)

    def decay_memory(self, base_decay=0.95, relevance_factor=1.05):
        current_time = time.time()
        for _, data in self.graph.nodes(data=True):
            time_diff = current_time - data['timestamp']
            decay_factor = base_decay ** time_diff
            adjusted_weight = data['weight'] * decay_factor * relevance_factor
            data['weight'] = adjusted_weight if adjusted_weight > 0.1 else 0.1

    def query_context(self, query_terms):
        relevant_texts = []
        for node, data in self.graph.nodes(data=True):
            if any(term in data['text'] for term in query_terms):
                relevant_texts.append((data['text'], data['weight']))
        relevant_texts.sort(key=lambda x: x[1], reverse=True)
        return relevant_texts[:3]

    def recall_context(self):
        texts = [data['text'] for node, data in self.graph.nodes(data=True) if data['type'] == 'user']
        return ' '.join(texts[-3:])

class ChatAI:
    def __init__(self):
        self.symbolic_rep = SymbolicRepresentation()
        self.embedding_model = embedding_functions.DefaultEmbeddingFunction()
        self.client = chromadb.PersistentClient(path="./chroma_client")
        self.collection = self.client.get_or_create_collection(name="collection_chat", embedding_function=self.embedding_model)
        self.memory_layer = GraphMemoryLayer()

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
        # Step 1: Convert the question and responses to symbolic representation
        question_symbols = set(self.symbolic_rep.text_to_symbol(question))
        response_symbols = [set(self.symbolic_rep.text_to_symbol(response)) for response in responses]

        # Step 2: Fetch context from the graph memory
        recent_context_symbols = set()
        for text, _ in self.memory_layer.query_context([question]):
            recent_context_symbols.update(self.symbolic_rep.text_to_symbol(text))
        
        # Step 3: Score responses based on their relevance to the question and context
        scores = []
        for response_set in response_symbols:
            relevance_to_question = len(response_set & question_symbols)
            relevance_to_context = len(response_set & recent_context_symbols)
            score = relevance_to_question + relevance_to_context
            scores.append(score)
        
        # Step 4: Select the response with the highest score
        best_response_index = scores.index(max(scores))
        return responses[best_response_index]

    def iterative_response_planning(self, question):
        # This method simulates generating a plan with multiple steps, revising each based on simulated feedback
        initial_responses = self.model(question, multi_response=True)
        # Evaluate the initial responses to select the most promising direction
        selected_response = self.evaluate_responses(question, initial_responses)
        # Simulate receiving new information or feedback that might affect the plan
        new_info = "Simulated new information"
        # Revise the plan based on the new information
        revised_response = self.model(new_info, multi_response=False)
        return revised_response
    
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

    def insert_chromadb(self):
        try:
            self.collection.delete(ids=["id1", "id2"])
            res = self.get_data_from_sqllite3()
            for c in res:
                em = self.embedding_model([c['conversation'], c['conversation']])
                self.collection.add(ids=[f"id_{c['conversation_id']}"], documents=[c['conversation']], embeddings=em, metadatas=[{"role": "conversation"}])
                print(f"Conversation {c['conversation_id']} inserted into chromadb successfully")
        except Exception as e:
            print(f"ChromaDB error: {e}")

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

chat = ChatAI()
chat.menu()


