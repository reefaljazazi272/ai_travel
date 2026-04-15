import streamlit as st
from main import get_recommendations

st.set_page_config(page_title="AI Travel Planner", page_icon="🌍")

st.title("🌍 AI Travel Recommendation System")
st.markdown("### Your AI-powered guide to the world")

query = st.text_input("Describe the trip you're dreaming of:", placeholder="e.g. I want to see mountains")

if st.button("Search for Destinations"):
    if query:
        with st.spinner('Searching...'):
            # طلب النتائج من دالة get_recommendations
            recs, scores = get_recommendations(query, top_n=3)
            
            if recs:
                st.write(f"### أفضل {len(recs)} وجهات مقترحة لك:")
                for i, (res, score) in enumerate(zip(recs, scores)):
                    # عرض كل وجهة في صندوق منفصل لمنع التكرار بصرياً
                    with st.expander(f"الخيار {i+1}: {res.name}, {res.country}", expanded=True):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**الوصف:** {res.description}")
                            st.info(f"الفئة: {res.category}")
                        with col2:
                            st.metric("نسبة التطابق", f"{score*100:.2f}%")
            else:
                st.error("لم يتم العثور على نتائج.")