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

# Symbolic Representation using spaCy for NLP
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
            if token.pos_ in ['NOUN', 'VERB']:
                symbols.append(f"{token.lemma_}:{token.pos_}")
        return symbols

# Graph-based Memory Layer enhanced with Symbolic Representation
class GraphMemoryLayer:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.symbolic_rep = SymbolicRepresentation()

    def update_graph(self, user_input, ai_response):
        current_time = time.time()
        user_symbols = self.symbolic_rep.text_to_symbol(user_input)
        ai_symbols = self.symbolic_rep.text_to_symbol(ai_response)
        
        user_node = f"user: {user_input}"
        ai_node = f"ai: {ai_response}"
        self.graph.add_node(user_node, type='user', text=user_input, symbols=user_symbols, timestamp=current_time)
        self.graph.add_node(ai_node, type='ai', text=ai_response, symbols=ai_symbols, timestamp=current_time)
        
        for symbol in set(user_symbols) & set(ai_symbols):
            self.graph.add_edge(user_node, ai_node, symbol=symbol, timestamp=current_time)

    def recall_context(self, current_input):
        decayed_text = []
        current_time = time.time()
        decay_threshold = 300  # Example: 5 minutes as decay threshold

        for node in self.graph:
            node_time = self.graph.nodes[node]['timestamp']
            if (current_time - node_time) < decay_threshold:
                decayed_text.append(self.graph.nodes[node]['text'])
        
        return ' '.join(decayed_text)

class ChatAI:
    def __init__(self):
        self.embedding_model = embedding_functions.DefaultEmbeddingFunction()
        self.client = chromadb.PersistentClient(path="./chroma_client")
        self.collection = self.client.get_or_create_collection(name="collection_chat", embedding_function=self.embedding_model)
        self.memory_layer = GraphMemoryLayer()

    def model(self, questions):
        try:
            llm = LlamaCpp(model_path="./mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                           n_ctx=32768,
                           n_threads=8,
                           n_gpu_layers=-1)
            conversation = ConversationChain(
                llm=llm,
                memory=ConversationBufferMemory()
            )
            
            context = self.memory_layer.recall_context(questions) + ' ' + questions
            response = conversation.predict(input=context).split("\n")[0]
            self.memory_layer.update_graph(questions, response)
            return response
        except Exception as e:
            print(f"Error during model prediction: {e}")
            return "Sorry, I encountered an error processing your request."

    def create_table(self):
        try:
            conn = sqlite3.connect('./chat_ai.db')
            cur = conn.cursor()
            sql = """CREATE TABLE IF NOT EXISTS chat_history_data (
                                        user varchar(8),
                                        ai varchar(8),
                                        user_word varchar(2000), 
                                        ai_word varchar(2000),
                                        updatetime varchar(64)
                                    );"""
            cur.execute(sql)
            conn.commit()
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
        finally:
            cur.close()
            conn.close()

    def insert_sqllite3(self, user_word, ai_word):
        try:
            conn = sqlite3.connect('./chat_ai.db')
            cur = conn.cursor()
            insert_sql = """insert into chat_history_data values(?,?,?,?,?) """
            cur.execute(insert_sql, ('user', 'assistant', user_word, ai_word, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
            conn.commit()
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
        finally:
            cur.close()
            conn.close()

    def get_data_from_sqllite3(self):
        try:
            conn = sqlite3.connect('./chat_ai.db')
            cur = conn.cursor()
            res = cur.execute("select * from chat_history_data")
            result_data = res.fetchall()
            return result_data
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return []
        finally:
            cur.close()
            conn.close()

    def insert_chromadb(self):
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
        st.title("Chat Bot")
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "Hi, how can I help you?"}]
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).markdown(msg["content"])
        if questions := st.chat_input():
            response = self.model(questions)
            st.session_state.messages.append({"role": "user", "content": questions})
            st.session_state.messages.append({"role": "assistant", "content": response})
            self.insert_sqllite3(user_word=questions, ai_word=response)
            # Trigger a rerun to update the UI with the new messages
            st.experimental_rerun()

        if st.button("Clear Chat"):
            st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you?"}]
            # Rerun after clearing chat to refresh the UI
            st.experimental_rerun()

    def menu(self):
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
        st.title("Chat History")
        res = self.get_data_from_sqllite3()
        for idx, (user, ai, user_word, ai_word, updatetime) in enumerate(res):
            st.text_area(label=f"Conversation {idx+1}", value=f"User: {user_word}\nAI: {ai_word}", height=100)

chat = ChatAI()
chat.menu()
