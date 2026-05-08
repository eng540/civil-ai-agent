import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai_tools import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
import os

st.set_page_config(page_title="Civil AI Agent", page_icon="🛠️")
st.title("🛠️ Civil Engineering AI Agent")
st.markdown("**وكيل ذكي متخصص في الهندسة المدنية**")

# ====================== API Keys ======================
gemini_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
tavily_key = st.secrets.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")

if not gemini_key or not tavily_key:
    st.error("❌ يرجى إضافة المفاتيح في Streamlit Secrets")
    st.stop()

# ====================== LLM + Tools ======================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # أو gemini-2.5-pro-exp إذا أردت قوة أكثر
    google_api_key=gemini_key,
    temperature=0.3
)

search_tool = TavilySearchResults(
    max_results=8,
    tavily_api_key=tavily_key
)

# ====================== الوكلاء ======================
researcher = Agent(
    role="باحث هندسة مدنية متخصص",
    goal="إجراء بحث عميق ودقيق في المواضيع الهندسية",
    backstory="أنت مهندس مدني خبير جداً، تبحث عن معلومات موثوقة ومعايير حديثة.",
    tools=[search_tool],
    llm=llm,
    verbose=False
)

writer = Agent(
    role="كاتب تقارير هندسية",
    goal="كتابة تقارير واضحة واحترافية",
    backstory="أنت كاتب تقارير فنية محترف، تكتب بأسلوب منظم وواضح.",
    llm=llm,
    verbose=False
)

# ====================== الواجهة ======================
topic = st.text_input("أدخل موضوع البحث الهندسي:", 
                     placeholder="مثال: تصميم أساسات مبنى في تربة رملية")

language = st.selectbox("اختر اللغة:", ["عربي", "English"])

if st.button("🚀 شغّل الوكيل", type="primary"):
    if not topic:
        st.warning("الرجاء إدخال الموضوع")
    else:
        with st.spinner("جاري البحث والكتابة... (قد يستغرق 40-90 ثانية)"):
            try:
                task1 = Task(
                    description=f"ابحث بشكل عميق عن: {topic}",
                    expected_output="ملخص بحث شامل",
                    agent=researcher
                )
                
                task2 = Task(
                    description=f"اكتب تقريراً هندسياً كاملاً ومنظماً باللغة {language} عن {topic}",
                    expected_output="تقرير Markdown احترافي",
                    agent=writer
                )
                
                crew = Crew(
                    agents=[researcher, writer],
                    tasks=[task1, task2],
                    process=Process.sequential,
                    verbose=1
                )
                
                result = crew.kickoff()
                
                st.success("✅ تم إنشاء التقرير بنجاح!")
                st.markdown(result)
                
            except Exception as e:
                st.error(f"حدث خطأ: {str(e)}")

st.caption("Powered by Gemini + Tavily + CrewAI")
