
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="technoAssistant",
    page_icon="🛠️",
    layout="wide"
)


STATUS_LABELS = {
    "supported": "Supported",
    "partially_supported": "Partially supported",
    "insufficient_context": "Insufficient context",
    "clarification_required": "Clarification required",
    "generation_failed": "Generation failed"
}


def check_backend():

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException:

        return None


def call_api(query):

    response = requests.post(
        f"{API_URL}/api/troubleshoot",
        json={"query": query},
        timeout=300
    )

    response.raise_for_status()

    return response.json()


def render_actions(title, items, numbered=False):

    if not items:
        return

    st.markdown(f"**{title}**")

    for index, item in enumerate(items, start=1):

        prefix = f"{index}." if numbered else "-"

        st.markdown(
            f"{prefix} {item['action']} "
            f"`{item['source_id']}`"
        )


def render_sources(sources):

    if not sources:
        return

    st.markdown("**Supporting knowledge base**")

    for source in sources:

        with st.expander(
            f"{source['id']} — {source['title']}"
        ):

            st.caption(
                f"Category: {source['category']}"
            )

            st.caption(
                "Reranker score: "
                f"{source['relevance_score']:.2f}"
            )

            content = source["content"]

            fields = [
                "description",
                "symptoms",
                "error_codes",
                "possible_causes",
                "diagnostic_checks",
                "known_resolutions",
                "escalation_team",
                "escalation_conditions",
                "required_information"
            ]

            for field in fields:

                value = content.get(field)

                if value:

                    label = field.replace(
                        "_", " "
                    ).title()

                    st.markdown(f"**{label}**")

                    for item in value.split(" | "):

                        st.write(item)


def render_issue(item, index):

    st.subheader(
        f"{index}. {item['issue']}"
    )

    status = item["status"]

    if status == "insufficient_context":

        st.warning(
            item.get(
                "message",
                "Insufficient knowledge-base information."
            )
        )

        return

    if status == "generation_failed":

        st.error(
            item.get(
                "message",
                "Answer generation failed."
            )
        )

        return

    answer = item.get("answer")

    if not answer:

        st.warning("No troubleshooting answer available.")

        return

    st.write(answer["summary"])

    render_actions(
        "Possible causes",
        answer["possible_causes"]
    )

    render_actions(
        "Diagnostic steps",
        answer["diagnostic_steps"],
        numbered=True
    )

    if answer["resolutions"]:

        st.markdown("**Recommended resolutions**")

        for resolution in answer["resolutions"]:

            with st.container(border=True):

                st.markdown(
                    f"**Condition:** "
                    f"{resolution['condition']}"
                )

                st.write(resolution["action"])

                st.caption(
                    f"Source: {resolution['source_id']}"
                )

    if answer["escalation"]:

        st.markdown("**Escalation guidance**")

        for escalation in answer["escalation"]:

            with st.container(border=True):

                st.markdown(
                    f"**Team:** {escalation['team']}"
                )

                st.write(escalation["condition"])

                st.caption(
                    f"Source: {escalation['source_id']}"
                )

    render_actions(
        "Information to collect",
        answer["required_information"]
    )

    render_sources(item.get("sources", []))


def render_response(result):

    status = result["status"]

    if status == "clarification_required":

        st.info(
            result.get(
                "message",
                "Please provide more details."
            )
        )

        return

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Issues identified",
        result.get("total_issues", 0)
    )

    col2.metric(
        "Supported issues",
        result.get("supported_issues", 0)
    )

    col3.metric(
        "Response time",
        f"{result['latency_seconds']:.2f}s"
    )

    if status == "supported":

        st.success("Troubleshooting guidance available.")

    elif status == "partially_supported":

        st.warning(
            "Some issues could not be answered "
            "using the knowledge base."
        )

    elif status == "insufficient_context":

        st.warning(
            "No sufficiently relevant knowledge "
            "base entries were found."
        )

    elif status == "generation_failed":

        st.error("Unable to generate a validated response.")

    for index, item in enumerate(
        result.get("issues", []),
        start=1
    ):

        st.divider()

        render_issue(item, index)


# Initialize chat history

if "messages" not in st.session_state:

    st.session_state.messages = []


# Sidebar

with st.sidebar:

    st.title("🛠️technoAssistant")

    st.caption("IT Operations Copilot")

    st.divider()

    if st.button(
        "New conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.subheader("System status")

    health = check_backend()

    if health:

        st.success("Backend online")

        st.metric(
            "Indexed runbooks",
            health["indexed_documents"]
        )

    else:

        st.error("Backend unavailable")

    st.divider()

    st.subheader("Technology")

    st.markdown("""
    - GPT-OSS-20B
    - Hybrid retrieval
    - Qdrant vector database
    - BM25 keyword search
    - Cross-encoder reranking
    """)

    st.divider()

    st.caption(
        "Prototype using simulated IT support "
        "runbooks. Verify guidance before "
        "performing operational changes."
    )


# Main interface

st.title("🛠️ technoAssistant")

st.caption(
    "Intelligent troubleshooting assistant "
    "for IT operations"
)


if not st.session_state.messages:

    st.markdown("### How can I help you troubleshoot?")

    st.write(
        "Describe an issue, error code or symptom "
        "to receive knowledge-base-backed guidance."
    )

    st.markdown("""
    **Example queries**

    - Payment API is returning HTTP 500
    - Database connection refused
    - DNS resolution failure
    - HTTP 500 and HTTP 505
    """)


# Render previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message["role"] == "user":

            st.write(message["content"])

        else:

            if "error" in message:

                st.error(message["error"])

            else:

                render_response(message["content"])


# Receive new user message

query = st.chat_input(
    "Describe your technical issue..."
)


if query:

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):

        st.write(query)

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "Analyzing issue and retrieving "
                "relevant knowledge..."
            ):

                result = call_api(query)

            render_response(result)

            st.session_state.messages.append({
                "role": "assistant",
                "content": result
            })

        except requests.RequestException:

            error = (
                "Unable to communicate with the "
                "troubleshooting backend. "
                "Ensure FastAPI is running."
            )

            st.error(error)

            st.session_state.messages.append({
                "role": "assistant",
                "error": error
            })
