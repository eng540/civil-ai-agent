# ======================================================
# Civil AI Agent
# Stable Streamlit Version
# Python 3.11
# ======================================================

# ======================================================
# SQLite Fix for ChromaDB on Streamlit Cloud
# ======================================================

try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

# ======================================================
# Imports
# ======================================================

import os
import traceback

import streamlit as st

from crewai import Agent
from crewai import Task
from crewai import Crew
from crewai import Process

from crewai_tools import tool
from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI

# ======================================================
# Streamlit Page Config
# ======================================================

st.set_page_config(
    page_title="Civil AI Agent",
    page_icon="🛠️",
    layout="wide"
)

# ======================================================
# Header
# ======================================================

st.title("🛠️ Civil Engineering AI Agent")
st.markdown("### وكيل ذكاء اصطناعي متخصص في الهندسة المدنية")

st.divider()

# ======================================================
# Sidebar
# ======================================================

with st.sidebar:
    st.header("⚙️ إعدادات النظام")

    model_name = st.selectbox(
        "اختر نموذج Gemini",
        [
            "gemini-2.5-flash",
            "gemini-2.5-pro"
        ],
        index=0
    )

    temperature = st.slider(
        "درجة الإبداع",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1
    )

    max_results = st.slider(
        "عدد نتائج البحث",
        min_value=3,
        max_value=10,
        value=5,
        step=1
    )

# ======================================================
# API Keys
# ======================================================

GEMINI_API_KEY = (
    st.secrets.get("GEMINI_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)

TAVILY_API_KEY = (
    st.secrets.get("TAVILY_API_KEY")
    or os.getenv("TAVILY_API_KEY")
)

# ======================================================
# Validate Keys
# ======================================================

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY غير موجود")
    st.info("أضف المفتاح داخل Streamlit Secrets")
    st.stop()

if not TAVILY_API_KEY:
    st.error("❌ TAVILY_API_KEY غير موجود")
    st.info("أضف المفتاح داخل Streamlit Secrets")
    st.stop()

# ======================================================
# Initialize LLM
# ======================================================

try:
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
        convert_system_message_to_human=True
    )

except Exception as e:
    st.error("❌ فشل تهيئة نموذج Gemini")
    st.exception(e)
    st.stop()

# ======================================================
# Initialize Search Tool
# ======================================================

try:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

@tool("Internet Search Tool")
def search_tool(query: str) -> str:
    \"\"\"
    Search the internet for engineering information.
    \"\"\"

    try:
        response = tavily_client.search(
            query=query,
            search_depth=\"advanced\",
            max_results=max_results
        )

        results = response.get(\"results\", [])

        if not results:
            return \"No results found.\"

        formatted = []

        for item in results:
            title = item.get(\"title\", \"\")\n
            content = item.get(\"content\", \"\")\n
            url = item.get(\"url\", \"\")\n
            formatted.append(
                f\"Title: {title}\\n\"\n
                f\"Content: {content}\\n\"\n
                f\"URL: {url}\\n\"\n
            )

        return \"\\n\\n\".join(formatted)

    except Exception as e:
        return f\"Search Error: {str(e)}\"
except Exception as e:
    st.error("❌ فشل تهيئة Tavily")
    st.exception(e)
    st.stop()

# ======================================================
# Agents
# ======================================================

researcher = Agent(
    role="باحث هندسة مدنية",
    goal="""
    إجراء بحث هندسي احترافي ودقيق
    وجمع معلومات موثوقة وحديثة
    حول المواضيع الهندسية المدنية.
    """,
    backstory="""
    مهندس مدني خبير في:
    - التصميم الإنشائي
    - التربة والأساسات
    - الخرسانة
    - الطرق والجسور
    - إدارة المشاريع
    - الكودات الهندسية
    """,
    tools=[search_tool],
    llm=llm,
    verbose=False,
    allow_delegation=False
)

writer = Agent(
    role="كاتب تقارير هندسية",
    goal="""
    تحويل نتائج البحث إلى تقارير هندسية
    احترافية ومنظمة وسهلة القراءة.
    """,
    backstory="""
    كاتب تقارير هندسية محترف
    متخصص في تبسيط المعلومات الفنية
    وكتابة تقارير احترافية.
    """,
    llm=llm,
    verbose=False,
    allow_delegation=False
)

reviewer = Agent(
    role="مراجع هندسي",
    goal="""
    مراجعة التقارير والتأكد من:
    - جودة المعلومات
    - صحة الصياغة
    - وضوح التقرير
    - التنظيم الاحترافي
    """,
    backstory="""
    خبير مراجعة هندسية
    يمتلك خبرة في التدقيق الفني.
    """,
    llm=llm,
    verbose=False,
    allow_delegation=False
)

# ======================================================
# Main Inputs
# ======================================================

col1, col2 = st.columns([3, 1])

with col1:
    topic = st.text_input(
        "أدخل موضوع البحث الهندسي",
        placeholder="مثال: تصميم أساسات مبنى في تربة رملية"
    )

with col2:
    language = st.selectbox(
        "لغة التقرير",
        [
            "العربية",
            "English"
        ]
    )

# ======================================================
# Extra Options
# ======================================================

include_solutions = st.checkbox(
    "تضمين الحلول الهندسية",
    value=True
)

include_materials = st.checkbox(
    "تضمين المواد المستخدمة",
    value=True
)

include_codes = st.checkbox(
    "تضمين الكودات والمعايير",
    value=True
)

# ======================================================
# Run Button
# ======================================================

st.divider()

run_button = st.button(
    "🚀 تشغيل الوكيل",
    type="primary",
    use_container_width=True
)

# ======================================================
# Execution
# ======================================================

if run_button:

    if not topic.strip():
        st.warning("⚠️ يرجى إدخال موضوع البحث")
        st.stop()

    with st.spinner("⏳ جاري تشغيل الوكلاء وتحليل البيانات..."):

        try:

            # ==================================================
            # Dynamic Instructions
            # ==================================================

            extra_requirements = []

            if include_solutions:
                extra_requirements.append("- قدم حلول هندسية عملية")

            if include_materials:
                extra_requirements.append("- اشرح المواد المستخدمة")

            if include_codes:
                extra_requirements.append("- اذكر الكودات والمعايير الهندسية")

            extra_text = "\n".join(extra_requirements)

            # ==================================================
            # Task 1 - Research
            # ==================================================

            research_task = Task(
                description=f"""
                قم بإجراء بحث هندسي احترافي حول:

                {topic}

                المطلوب:
                - شرح الفكرة الهندسية
                - أفضل الممارسات
                - المشاكل الشائعة
                - طرق التنفيذ
                - أحدث التقنيات

                {extra_text}
                """,

                expected_output="""
                تقرير بحث هندسي احترافي
                يحتوي على معلومات دقيقة ومنظمة.
                """,

                agent=researcher
            )

            # ==================================================
            # Task 2 - Writing
            # ==================================================

            writing_task = Task(
                description=f"""
                اكتب تقريراً احترافياً باللغة {language}
                اعتماداً على نتائج البحث.

                يجب أن يكون التقرير:
                - منظم
                - واضح
                - احترافي
                - سهل القراءة
                - يحتوي عناوين فرعية
                """,

                expected_output="""
                تقرير هندسي احترافي كامل.
                """,

                agent=writer
            )

            # ==================================================
            # Task 3 - Review
            # ==================================================

            review_task = Task(
                description="""
                راجع التقرير النهائي:
                - صحح الأخطاء
                - حسن الصياغة
                - تأكد من جودة المعلومات
                - اجعل التقرير احترافياً
                """,

                expected_output="""
                نسخة نهائية احترافية من التقرير.
                """,

                agent=reviewer
            )

            # ==================================================
            # Crew
            # ==================================================

            crew = Crew(
                agents=[
                    researcher,
                    writer,
                    reviewer
                ],
                tasks=[
                    research_task,
                    writing_task,
                    review_task
                ],
                process=Process.sequential,
                verbose=False
            )

            # ==================================================
            # Run
            # ==================================================

            result = crew.kickoff()

            # ==================================================
            # Output
            # ==================================================

            st.success("✅ تم إنشاء التقرير بنجاح")

            st.divider()

            st.markdown("# 📄 التقرير النهائي")

            st.markdown(str(result))

        except Exception as e:

            st.error("❌ حدث خطأ أثناء تشغيل النظام")

            st.code(traceback.format_exc())

# ======================================================
# Footer
# ======================================================

st.divider()

st.caption("Civil AI Agent • Powered by CrewAI + Gemini + Tavily")