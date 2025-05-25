import streamlit as st
import traceback

# Import với error handling
try:
    from app.rag_pipeline import load_rag_chain

    IMPORT_SUCCESS = True
except Exception as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("🧠 RAG Chatbot - Healthcare Assistant")

# Kiểm tra import
if not IMPORT_SUCCESS:
    st.error(f"❌ Lỗi import: {IMPORT_ERROR}")
    st.info("Vui lòng kiểm tra lại file app/rag_pipeline.py")
    st.stop()

# Danh sách model
model_options = ["llama3", "mistral", "deepseek"]
selected_model = st.selectbox("Chọn mô hình LLM", model_options)

# Debug: Hiển thị trạng thái
with st.expander("🔍 Debug Info", expanded=False):
    st.write("Session State Keys:", list(st.session_state.keys()))
    st.write("Selected Model:", selected_model)

# Khởi tạo session state đơn giản
if "messages" not in st.session_state:
    st.session_state.messages = {model: [] for model in model_options}

if "chains_loaded" not in st.session_state:
    st.session_state.chains_loaded = {}

if "current_model" not in st.session_state:
    st.session_state.current_model = selected_model


# Lazy loading - chỉ load khi cần
def get_chain(model_name):
    if model_name not in st.session_state.chains_loaded:
        try:
            with st.spinner(f"Đang tải model {model_name}..."):
                st.session_state.chains_loaded[model_name] = load_rag_chain(model_name)
                st.success(f"✅ Đã tải model {model_name}")
        except Exception as e:
            st.error(f"❌ Lỗi khi tải model {model_name}: {str(e)}")
            st.code(traceback.format_exc())
            return None
    return st.session_state.chains_loaded.get(model_name)


# Cập nhật model hiện tại
if selected_model != st.session_state.current_model:
    st.session_state.current_model = selected_model

# Hiển thị lịch sử chat
current_messages = st.session_state.messages[selected_model]
for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Thêm nút test
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("🧪 Test Model"):
        chain = get_chain(selected_model)
        if chain:
            st.success("Model hoạt động bình thường!")
        else:
            st.error("Model không thể tải!")

with col2:
    if st.button("🗑️ Xóa lịch sử"):
        st.session_state.messages[selected_model] = []
        st.rerun()

with col3:
    if st.button("🔄 Reset App"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Chat input
if prompt := st.chat_input("Nhập câu hỏi về chăm sóc sức khỏe..."):
    # Thêm message user
    st.session_state.messages[selected_model].append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Tạo response
    with st.chat_message("assistant"):
        try:
            chain = get_chain(selected_model)
            if chain is None:
                response = "❌ Model chưa được tải thành công. Vui lòng thử lại."
            else:
                with st.spinner("Đang xử lý..."):
                    response = chain.run(prompt)
        except Exception as e:
            response = f"❌ Lỗi: {str(e)}"
            st.code(traceback.format_exc())

        st.markdown(response)

        # Lưu response
        st.session_state.messages[selected_model].append({
            "role": "assistant",
            "content": response
        })

# Hiển thị thông tin hệ thống
st.sidebar.title("📊 System Info")
st.sidebar.write(f"**Current Model:** {selected_model}")
st.sidebar.write(f"**Messages:** {len(current_messages)}")
st.sidebar.write(f"**Loaded Models:** {list(st.session_state.chains_loaded.keys())}")