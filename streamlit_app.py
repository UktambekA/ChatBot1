import streamlit as st
from book_chatbot import BookChatbot
from utils import save_qa_history_pdf

st.set_page_config(
    page_title="Kitob Chatbot",
    page_icon="📖",
    layout="wide",
)

qa_history = st.session_state.get("qa_history", [])

st.title("📖 Kitob Chatbot")
st.markdown("**Istalgan kitob asosida savol-javob qilish, test yaratish yoki xulosa olish imkoniyati.**")

openai_api_key = st.text_input("OpenAI API kalitini kiriting:", type="password")
pdf_file = st.file_uploader("PDF faylni yuklang:", type=["pdf"])

if pdf_file and openai_api_key:
    temp_pdf_path = f"./uploaded_{pdf_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_file.read())

    chatbot = BookChatbot(pdf_path=temp_pdf_path, openai_api_key=openai_api_key)

    st.info("PDF fayli o'qilmoqda...")
    pdf_text = chatbot.extract_text_from_pdf(pdf_file)
    documents = chatbot.split_text(pdf_text)
    vector_store = chatbot.create_vector_store(documents)
    st.success("PDF fayli muvaffaqiyatli yuklandi!")

    max_tokens = st.slider("Maksimal javob uzunligi (so'zlar soni):", min_value=100, max_value=500, value=300)
    rag_chain = chatbot.create_rag_chain(vector_store, max_tokens)

    st.markdown("## 🗨️ Savol bering")
    user_question = st.text_area("Savolingizni kiriting:", "")
    if st.button("Javobni olish"):
        if user_question.strip():
            with st.spinner("Javob tayyorlanmoqda..."):
                answer = chatbot.answer_question(rag_chain, user_question)
            st.markdown(f"**Javob:** {answer}")

            qa_history.append({"Savol": user_question, "Javob": answer})
            st.session_state.qa_history = qa_history
        else:
            st.warning("Iltimos, savol kiriting.")

    if qa_history:
        st.markdown("## 📝 Savol-Javob Tarixi")
        for i, qa in enumerate(qa_history):
            st.markdown(f"**{i + 1}. Savol:** {qa['Savol']}")
            st.markdown(f"**Javob:** {qa['Javob']}")
            st.markdown("---")

        pdf_path = save_qa_history_pdf(qa_history)
        if pdf_path:
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📥 Tarixni PDF formatda yuklab olish",
                    data=pdf_file,
                    file_name="qa_history.pdf",
                    mime="application/pdf",
                )
        else:
            st.error("PDF fayl yaratishda xatolik yuz berdi.")

st.markdown("## 🔄 Yangi PDF yuklash")
if st.button("Qayta yuklash"):
    st.session_state.clear()
    st.experimental_rerun()
