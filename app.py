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
        symbols = [f"{token.lemma_}:{token.pos_}" for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'NUM', 'SYM', 'ADP', 'CONJ', 'DET', 'PART', 'PUNCT', 'X']]
        return symbols

class GraphMemoryLayer:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.symbolic_rep = SymbolicRepresentation()

    def update_graph(self, user_input, ai_response):
        user_symbols = self.symbolic_rep.text_to_symbol(user_input)
        ai_symbols = self.symbolic_rep.text_to_symbol(ai_response)
        
        user_node = f"user_{len(self.graph.nodes) + 1}"
        ai_node = f"ai_{len(self.graph.nodes) + 2}"
        self.graph.add_node(user_node, type='user', text=user_input, symbols=user_symbols)
        self.graph.add_node(ai_node, type='ai', text=ai_response, symbols=ai_symbols)
        
        for symbol in set(user_symbols) & set(ai_symbols):
            self.graph.add_edge(user_node, ai_node, symbol=symbol)

    def recall_context(self):
        texts = [data['text'] for node, data in self.graph.nodes(data=True) if data['type'] == 'user']
        return ' '.join(texts[-3:])

class ChatAI:
    def __init__(self):
        self.embedding_model = embedding_functions.DefaultEmbeddingFunction()
        self.client = chromadb.PersistentClient(path="./chroma_client")
        self.collection = self.client.get_or_create_collection(name="collection_chat", embedding_function=self.embedding_model)
        self.memory_layer = GraphMemoryLayer()

    def model(self, question):
        try:
            llm = LlamaCpp(model_path="./mistral-7b-instruct-v0.2.Q4_K_M.gguf", n_ctx=32768, n_threads=8, n_gpu_layers=-1)
            conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory())
            
            context = self.memory_layer.recall_context() + ' ' + question
            response = conversation.predict(input=context).split("\n")[0]
            
            self.memory_layer.update_graph(question, response)
            
            return response
        except Exception as e:
            print(f"Error during model prediction: {e}")
            return "Sorry, I encountered an error processing your request."

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
        """
        Inserts chat history data into ChromaDB.
        """
        try:
            self.collection.delete(ids=["id1", "id2"])
            res = self.get_data_from_sqllite3()
            for c in res:
                em = self.embedding_model([c[2], c[3]])
                self.collection.add(ids=['id1', 'id2'], documents=[c[2], c[3]], embeddings=em, metadatas=[{"role": "user"}, {"role": "assistant"}])
                print(f"{c[2]} and {c[3]} inserted into chromadb successfully")
        except Exception as e:
            print(f"ChromaDB error: {e}")

    def chatbot_ui(self):
        """
        Implements the user interface for the chatbot.
        """
        st.title("Chat Bot")
        # Use a session variable to keep track of a list of messages as a conversation
        if "conversation" not in st.session_state:
            st.session_state["conversation"] = []
        
        for msg in st.session_state["conversation"]:
            st.chat_message(msg["role"]).markdown(msg["content"])

        if questions := st.chat_input():
            response = self.model(questions)
            st.session_state["conversation"].append({"role": "user", "content": questions})
            st.session_state["conversation"].append({"role": "assistant", "content": response})
            # Do not insert here, we will insert the whole conversation when it ends
            st.rerun()

        if st.button("End Conversation"):
            # Insert the whole conversation into the database
            self.insert_sqllite3(st.session_state["conversation"])
            st.session_state["conversation"] = []
            st.rerun()

        # Additional button to start a new conversation, not necessary if using End Conversation
        if st.button("Start New Conversation"):
            if st.session_state["conversation"]:
                # Insert the current conversation before starting a new one
                self.insert_sqllite3(st.session_state["conversation"])
            st.session_state["conversation"] = []
            st.rerun()


    def menu(self):
        """
        Displays the menu options and handles user's choice.
        """
        self.create_table()
        menu = ['Home', 'Chat Bot', 'Chat History']
        choice = st.sidebar.radio('Menu', menu)
        if choice == 'Home':
            st.title("Welcome to Chat Bot!")
            st.text("I am a chat bot! I can help you answer some questions that you want to know. Let’s begin!")
            image_path = "./chat_bot.png"  # Ensure this path is correct
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
        """
        Displays the chat history stored in SQLite database.
        """
        st.title("Chat History")
        res = self.get_data_from_sqllite3()
        for idx, item in enumerate(res):
            conversation_id = item['conversation_id']
            conversation = item['conversation']
            updatetime = item['updatetime']
            # Assuming the conversation is a list of messages
            conversation_text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in conversation)
            st.text_area(label=f"Conversation {conversation_id} from {updatetime}", value=conversation_text, height=200)

chat = ChatAI()
chat.menu()
