import os
import hashlib
import logging
import shutil
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


class BookChatbot:
    def __init__(self, pdf_path, openai_api_key, model="gpt-4o-mini"):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.pdf_path = pdf_path
        self.model = model
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def _get_pdf_hash(self):
        """PDF faylining xeshini yaratadi."""
        try:
            with open(self.pdf_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except IOError as e:
            self.logger.error(f"PDFni o'qishda xatolik: {e}")
            raise

    def _check_cached_vector_store(self, file_hash):
        """Keshlangan vektorlarni yuklaydi."""
        cache_path = f"vector_store_{file_hash}"
        if os.path.exists(cache_path):
            try:
                return FAISS.load_local(cache_path, self.embeddings)
            except Exception as e:
                self.logger.warning(f"Keshlangan vektor yuklashda xatolik: {e}")
        return None

    def _save_vector_store(self, vector_store, file_hash):
        """Vektor do'konini saqlaydi."""
        cache_path = f"vector_store_{file_hash}"
        try:
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path)

            vector_store.save_local(cache_path)
            self.logger.info(f"Vektor {cache_path}ga saqlandi.")
        except Exception as e:
            self.logger.error(f"Vektor saqlashda xatolik: {e}")

    def extract_text_from_pdf(self, text):
        """PDF faylidan barcha matnni birlashtiradi."""
        from PyPDF2 import PdfReader
        reader = PdfReader(self.pdf_path)
        text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
        return text

    def split_text(self, text, chunk_size=1000, chunk_overlap=200):
        """Matnni bo'laklarga ajratadi."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return text_splitter.split_text(text)

    def create_vector_store(self, documents):
        """Vektor do'konini yaratadi."""
        file_hash = self._get_pdf_hash()
        cached_store = self._check_cached_vector_store(file_hash)
        if cached_store:
            return cached_store

        vector_store = FAISS.from_texts(documents, self.embeddings)
        self._save_vector_store(vector_store, file_hash)
        return vector_store

    def create_rag_chain(self, vector_store, max_tokens):
        """RAG chain yaratadi."""
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 5}
        )

        prompt = ChatPromptTemplate.from_template("""
        Quyidagi kontekst berilgan kitobdan olingan. Iltimos, foydalanuvchi so'roviga mos ravishda harakat qiling:

        1. Agar foydalanuvchi savol bersa - kontekstdan foydalanib javob bering.
        2. Agar foydalanuvchi test/savollar tuzishni so'rasa - kontekstdagi ma'lumotlar asosida test tuzing.
        3. Agar foydalanuvchi xulosa so'rasa - qisqacha xulosani taqdim qiling.
        4. Aloqador bo'lmagan so'rov uchun mos javob bering.

        Kontekst: {context}
        Foydalanuvchi so'rovi: {question}

        Javob:
        """)

        llm = ChatOpenAI(
            model_name=self.model,
            temperature=0.3,
            max_tokens=max_tokens
        )

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain

    def answer_question(self, rag_chain, question):
        """Foydalanuvchi savollariga javob beradi."""
        try:
            answer = rag_chain.invoke(question)
            if not answer or "Kechirasiz" in answer:
                return "Kechirasiz, bu so'rov kitob mavzusiga aloqador emas."
            return answer
        except Exception as e:
            self.logger.error(f"Savolga javob berishda xatolik: {e}")
            return "Kechirasiz, savolga javob berishda xatolik yuz berdi."
