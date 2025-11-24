import json

import requests
import streamlit as st

# Page config
st.set_page_config(page_title="Transcript Scorer", page_icon="📝", layout="wide")

# Title and description
st.title("📝 Transcript Scoring Tool")
st.markdown("""
Score self-introduction transcripts using AI-powered rubric evaluation.
Upload a transcript and get detailed feedback on communication skills.
""")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")
api_url = st.sidebar.text_input(
    "API Endpoint", value="http://localhost:8000/score", help="FastAPI backend URL"
)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📄 Transcript Input")

    # Input method selection
    input_method = st.radio(
        "Choose input method:", ["✍️ Text Area", "📁 Upload File"], horizontal=True
    )

    transcript_text = None
    transcript_file = None

    if input_method == "✍️ Text Area":
        transcript_text = st.text_area(
            "Paste your transcript here:",
            height=250,
            placeholder="Hello everyone, my name is...",
            help="Enter the spoken transcript text",
        )
    else:
        transcript_file = st.file_uploader(
            "Upload transcript file:",
            type=["txt", "pdf"],
            help="Upload a TXT or PDF file containing the transcript",
        )

with col2:
    st.subheader("⚙️ Optional Settings")

    # Duration input
    duration = st.number_input(
        "Duration (seconds)",
        min_value=0,
        value=0,
        help="Speech duration for WPM calculation (optional)",
    )

    # Custom rubric option
    st.markdown("---")
    use_custom_rubric = st.checkbox(
        "📋 Use Custom Rubric",
        help="Upload your own rubric instead of using the default",
    )

    rubric_file = None
    if use_custom_rubric:
        rubric_file = st.file_uploader(
            "Upload rubric:",
            type=["xlsx", "json"],
            help="Upload custom rubric (Excel or JSON format)",
        )

# Score button
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])

with col_btn1:
    score_button = st.button(
        "🚀 Score Transcript", type="primary", use_container_width=True
    )

with col_btn2:
    clear_button = st.button("🔄 Clear", use_container_width=True)

# Clear functionality
if clear_button:
    st.rerun()

# Scoring logic
if score_button:
    # Validation
    if not transcript_text and not transcript_file:
        st.error("❌ Please provide a transcript (text or file)")
    else:
        with st.spinner("🔄 Analyzing transcript..."):
            try:
                # Prepare request
                files = {}
                data = {}

                # Add transcript
                if transcript_text:
                    data["transcript"] = transcript_text
                elif transcript_file:
                    files["transcript_file"] = (
                        transcript_file.name,
                        transcript_file.getvalue(),
                        transcript_file.type,
                    )

                # Add duration if provided
                if duration > 0:
                    data["duration_seconds"] = duration

                # Add custom rubric if provided
                if rubric_file:
                    files["rubric_file"] = (
                        rubric_file.name,
                        rubric_file.getvalue(),
                        rubric_file.type,
                    )

                # Make API request
                response = requests.post(api_url, data=data, files=files)

                if response.status_code == 200:
                    result = response.json()

                    # Display results
                    st.success("✅ Scoring Complete!")

                    # Overall score section
                    st.markdown("---")
                    st.subheader("📊 Overall Results")

                    col_score1, col_score2, col_score3 = st.columns(3)

                    with col_score1:
                        st.metric(
                            "Overall Score",
                            f"{result['overall_score']:.1f}/100",
                            delta=None,
                        )

                    with col_score2:
                        st.metric("Word Count", result["word_count"])

                    with col_score3:
                        # Calculate grade
                        score = result["overall_score"]
                        if score >= 90:
                            grade = "A+"
                            color = "🟢"
                        elif score >= 80:
                            grade = "A"
                            color = "🟢"
                        elif score >= 70:
                            grade = "B"
                            color = "🟡"
                        elif score >= 60:
                            grade = "C"
                            color = "🟠"
                        else:
                            grade = "D"
                            color = "🔴"

                        st.metric("Grade", f"{color} {grade}")

                    # Progress bar
                    st.progress(result["overall_score"] / 100)

                    # Detailed breakdown
                    st.markdown("---")
                    st.subheader("📋 Detailed Breakdown")

                    # Display per-criterion results
                    for idx, criterion in enumerate(result["per_criterion"], 1):
                        with st.expander(
                            f"{idx}. {criterion['criterion']} - {criterion['metric']} "
                            f"({criterion['score']:.1f}/{criterion['max_score']:.1f})",
                            expanded=(idx == 1),
                        ):
                            # Score progress
                            score_pct = (
                                (criterion["score"] / criterion["max_score"] * 100)
                                if criterion["max_score"] > 0
                                else 0
                            )
                            st.progress(score_pct / 100)

                            # Feedback
                            st.markdown("**💬 Feedback:**")
                            st.info(criterion["feedback"])

                            # Details
                            if criterion.get("details"):
                                st.markdown("**🔍 Details:**")

                                details = criterion["details"]

                                # Keywords found
                                if details.get("keywords_found"):
                                    st.markdown(
                                        f"**Keywords Found:** {', '.join(details['keywords_found'])}"
                                    )

                                # Calculated value
                                if details.get("calculated_value") is not None:
                                    st.markdown(
                                        f"**Calculated Value:** {details['calculated_value']}"
                                    )

                                # Reasoning
                                if details.get("reasoning"):
                                    st.markdown(
                                        f"**Reasoning:** {details['reasoning']}"
                                    )

                    # Download results
                    st.markdown("---")
                    st.download_button(
                        label="📥 Download Results (JSON)",
                        data=json.dumps(result, indent=2),
                        file_name="transcript_score.json",
                        mime="application/json",
                    )

                else:
                    st.error(f"❌ Error: {response.status_code}")
                    st.json(response.json())

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Cannot connect to API. Make sure the FastAPI server is running on http://localhost:8000"
                )
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: gray;'>
    <small>Powered by Google Gemini AI | Built with Streamlit & FastAPI</small>
</div>
""",
    unsafe_allow_html=True,
)
