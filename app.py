import streamlit as st
from orchestrator import Orchestrator
from dotenv import load_dotenv
from datetime import datetime

# Load environment
load_dotenv()

# Page config
st.set_page_config(
    page_title="PartSelect Assistant",
    page_icon="./database/images/icon.png",
    layout="wide"
)

# --------------------------
# Branding CSS
# --------------------------
st.markdown("""
<style>

:root {
    --ps-yellow: #FFCC00;
    --ps-green: #004D4D;
}

/* Main App */
body, .stApp {
    background-color: white !important;
    color: var(--ps-green);
    font-family: "Arial", sans-serif;
}

/* Buttons */
.stButton>button {
    background-color: var(--ps-yellow) !important;
    color: var(--ps-green) !important;
    border-radius: 6px;
    border: 1px solid var(--ps-green);
    font-weight: 600;
}

.stButton>button:hover {
    background-color: #e6b800 !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: var(--ps-green) !important;
}

/* Chat message borders */
.stChatMessage > div {
    border: 1.5px solid var(--ps-green) !important;
    border-radius: 8px !important;
    padding: 10px !important;
}

/* Sidebar-style boxes */
.sidebar-box {
    background: white;
    border: 2px solid var(--ps-green);
    padding: 12px;
    border-radius: 6px;
    margin-bottom: 15px;
}

.contact-box {
    background: #FFF8D1;
    border: 2px solid var(--ps-green);
    padding: 14px;
    border-radius: 6px;
    margin-bottom: 20px;
    color: var(--ps-green);
}

.cart-container {
    background: white;
    border: 2px solid var(--ps-green);
    padding: 14px;
    border-radius: 6px;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------
# Initialize session state
# --------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.orchestrator = Orchestrator()
    st.session_state.cart = []

    greeting = """
Welcome to the PartSelect Assistant.

I can help you with:
- Finding and comparing appliance parts 
- Troubleshooting and identifying issues  
- Repair guides and videos  
- Building your parts list  
"""
    st.session_state.messages.append({"role": "assistant", "content": greeting})

# --------------------------
# Helper functions
# --------------------------
def add_to_cart(product):

    if not any(item['id'] == product['id'] for item in st.session_state.cart):
        st.session_state.cart.append(product)
        st.success(f"Added {product['name']}")
    else:
        st.info("Already in cart")


def remove_from_cart(product_id):
    st.session_state.cart = [
        item for item in st.session_state.cart if item['id'] != product_id
    ]
    st.success("Removed from cart")


def export_chat_history():
    export_text = "# PartSelect Chat History\n\n"
    export_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        export_text += f"## {role}\n{msg['content']}\n\n"

        if msg.get('products'):
            export_text += "### Products:\n"
            for prod in msg['products']:
                export_text += f"- {prod['name']} (${prod['price']})\n  {prod.get('url','N/A')}\n"
            export_text += "\n"

    if st.session_state.cart:
        export_text += "\n## Shopping Cart\n\n"
        for item in st.session_state.cart:
            export_text += f"- {item['name']} - ${item['price']}\n  {item['url']}\n"

    return export_text


def end_chat():
    st.session_state.messages = []
    st.session_state.cart = []
    st.session_state.orchestrator.reset_conversation()
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Chat reset. How can I assist you?"
    })


# --------------------------
# Layout: Main Chat + Info Panel
# --------------------------
col_main, col_info = st.columns([3, 1])

# ==========================
# LEFT COLUMN (CHAT)
# ==========================
with col_main:

    # Logo & title
    st.image("./database/images/ps-header-logo-here-to-help.svg", width=220)
    st.title("PartSelect Assistant")
    st.markdown("""
<div class="contact-box">
<b>Customer Service:</b> 1-866-319-8402  
<b>Hours:</b> Monday to Saturday, 8am – 8pm EST
</div>
""", unsafe_allow_html=True)

    # -------------------------
    # Chat container (scrollable)
    # -------------------------
    chat_container = st.container()

    def render_message(message, idx):
        """Render a single message with products and videos."""
        role = message.get("role", "assistant")  # fallback to assistant
        with st.chat_message(role):
            st.markdown(message.get("content", ""))

            # Products
            for p_idx, product in enumerate(message.get("products", [])):
                with st.container():
                    p1, p2 = st.columns([4, 1])
                    with p1:
                        st.markdown(f"**{product['name']}**")
                        st.caption(f"${product.get('price','N/A')}")
                        if product.get('brand'):
                            st.caption(f"Brand: {product['brand']}")
                    with p2:
                        if st.button("Add to Cart", key=f"add_{product['id']}_{idx}_{p_idx}"):
                            add_to_cart(product)
                            st.rerun()

            # Videos
            for vid in message.get("videos", []):
                vid_url = vid["url"] if isinstance(vid, dict) else vid
                if vid_url:
                    st.video(vid_url)

    # Render all saved messages
    for idx, message in enumerate(st.session_state.messages):
        render_message(message, idx)

    # Render pending assistant message if exists
    if "pending_assistant" in st.session_state:
        render_message(st.session_state.pending_assistant, -1)

    # -------------------------
    # Chat input
    # -------------------------
    if prompt := st.chat_input("Ask about parts or troubleshooting..."):
        # Append user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process assistant response
        with st.spinner("Processing..."):
            response, products, media = st.session_state.orchestrator.process_query(prompt)

            # Save pending assistant message
            st.session_state.pending_assistant = {
                "role": "assistant",
                "content": response,
                "products": products or [],
                "videos": media.get("videos", []) if media else []
            }

        # Immediately rerun to display pending message
        st.rerun()

    # -------------------------
    # Confirm assistant message and move from pending to messages
    # -------------------------
    if "pending_assistant" in st.session_state:
        st.session_state.messages.append(st.session_state.pending_assistant)
        del st.session_state.pending_assistant

    # -------------------------
    # Contact info above cart
    # -------------------------
    st.divider()

    # ==========================
    # CART
    # ==========================
    st.subheader("Shopping Cart")
    if st.session_state.cart:
        total = 0
        for item in st.session_state.cart:
            with st.container():
                st.markdown(f"**{item['name'][:40]}**")
                st.caption(f"${item.get('price','N/A')}")
                c1, c2 = st.columns([1,1])

                with c1:
                    if st.button("Remove", key=f"rm_{item['id']}"):
                        remove_from_cart(item['id'])
                        st.rerun()
                with c2:
                    if item.get("url"):
                        st.markdown(f"[View Part]({item['url']})")

                try:
                    total += float(item.get("price", 0))
                except:
                    pass

        st.markdown(f"### Total: ${total:.2f}")
    else:
        st.info("Cart empty")

    # Footer buttons
    st.divider()
    b2, = st.columns(1)

    with b2:
        if st.button("End Chat", use_container_width=True):
            end_chat()
            st.rerun()

# ==========================
# RIGHT COLUMN (INFO PANEL)
# ==========================
with col_info:
    st.subheader("Customer Service Links")
    st.markdown("""
<div class="sidebar-box">
<b>Contact Us</b><br>
Full customer service options. <br>
<a href="https://www.partselect.com/Contact" target="_blank">Visit Contact Page</a>
                
<hr>

<b>Self Service</b><br>
Manage orders, track packages, and submit returns. <br>
<a href="https://www.partselect.com/user/self-service/" target="_blank">Go to Self Service</a>

<hr>

<b>Order by Phone</b><br>
Toll-free: 1-866-319-8402  
8am–8pm EST (Mon–Sat)

<hr>

<b>Site Use</b><br>
<a href="mailto:CustomerService@PartSelect.com">CustomerService@PartSelect.com</a>

<b>Location</b><br>
Ships from 30+ U.S. warehouses.
</div>
""", unsafe_allow_html=True)


   