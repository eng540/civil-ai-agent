# ====================== حل ChromaDB على Streamlit ======================
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# =====================================================================

import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai_tools import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
import os

st.set_page_config(page_title="Civil AI Agent", page_icon="🛠️")
st.title("🛠️ Civil Engineering AI Agent")
st.markdown("**وكيل ذكي متخصص في الهندسة المدنية**")

# API Keys
gemini_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
tavily_key = st.secrets.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")

if not gemini_key or not tavily_key:
    st.error("❌ يرجى إضافة المفاتيح في Secrets")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.3
)

search_tool = TavilySearchResults(max_results=6, tavily_api_key=tavily_key)

# الوكلاء
researcher = Agent(
    role="باحث هندسة مدنية",
    goal="بحث دقيق ومفيد",
    backstory="مهندس مدني خبير",
    tools=[search_tool],
    llm=llm,
    verbose=False
)

writer = Agent(
    role="كاتب تقارير",
    goal="كتابة تقرير واضح",
    backstory="كاتب تقارير هندسية",
    llm=llm,
    verbose=False
)

topic = st.text_input("أدخل موضوع البحث:", placeholder="تصميم أساسات مبنى في تربة رملية")
language = st.selectbox("اللغة:", ["عربي", "English"])

if st.button("🚀 شغّل الوكيل", type="primary"):
    if topic:
        with st.spinner("جاري العمل... قد يستغرق 40-80 ثانية"):
            try:
                task1 = Task(description=f"ابحث عن: {topic}", expected_output="ملخص بحث", agent=researcher)
                task2 = Task(description=f"اكتب تقرير باللغة {language} عن {topic}", expected_output="تقرير كامل", agent=writer)
                
                crew = Crew(agents=[researcher, writer], tasks=[task1, task2], process=Process.sequential)
                result = crew.kickoff()
                
                st.success("✅ تم بنجاح!")
                st.markdown(result)
            except Exception as e:
                st.error(f"خطأ: {str(e)}")
    else:
        st.warning("أدخل الموضوع أولاً")
