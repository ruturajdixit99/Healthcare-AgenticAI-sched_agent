# intelligent_healthcare_app.py
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict
import json
from datetime import datetime

# ----------------------------
# State Schema Definition
# ----------------------------
class AppState(TypedDict):
    input: str
    output: str
    route: str
    context: str

# ----------------------------
# Initialize LLM
# ----------------------------
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4", temperature=0.3, openai_api_key=api_key)

# ----------------------------
# INTELLIGENT SYMPTOM ANALYZER
# ----------------------------
def intelligent_symptom_analyzer(symptoms: str) -> str:
    """
    Intelligent symptom analysis using LLM - handles ANY symptoms
    """
    system_prompt = """You are an experienced medical assistant with extensive knowledge of symptoms, conditions, and medical protocols. 

    Your role is to:
    1. Analyze patient symptoms comprehensively
    2. Identify possible conditions or diagnoses
    3. Assess severity level (mild, moderate, severe, emergency)
    4. Provide appropriate recommendations
    5. Determine urgency of medical attention needed

    Guidelines:
    - Be thorough but not alarmist
    - Consider multiple possibilities
    - Provide clear, understandable explanations
    - Always recommend professional medical consultation
    - Include red flags that require immediate attention
    - Be empathetic and supportive

    Format your response with:
    - **Possible Conditions:** List likely diagnoses
    - **Severity Assessment:** Rate the urgency
    - **Immediate Actions:** What to do now
    - **When to Seek Help:** Timeline for medical attention
    - **Red Flags:** Emergency warning signs
    """
    
    user_prompt = f"""
    Patient presents with the following symptoms: {symptoms}
    
    Please provide a comprehensive analysis including possible conditions, severity assessment, 
    immediate recommendations, and guidance on when to seek medical attention.
    
    Consider various possibilities and provide helpful, actionable advice while emphasizing 
    the importance of professional medical evaluation.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm(messages)
        return response.content
    except Exception as e:
        return f"I apologize, but I'm having trouble analyzing your symptoms right now. Please consult with a healthcare professional for proper evaluation. Error: {str(e)}"

# ----------------------------
# INTELLIGENT APPOINTMENT SCHEDULER
# ----------------------------
def intelligent_appointment_scheduler(request: str) -> str:
    """
    Intelligent appointment booking that understands natural language requests
    """
    system_prompt = """You are a professional appointment scheduling assistant for a healthcare facility.

    Your capabilities:
    1. Parse natural language appointment requests
    2. Extract relevant information (name, doctor, specialty, time preferences)
    3. Suggest available slots based on preferences
    4. Handle appointment modifications and cancellations
    5. Provide appointment confirmations and reminders

    Available doctors and specialties:
    - Dr. Sarah Johnson (General Practice) - Available: Mon-Fri 9AM-5PM
    - Dr. Michael Chen (Cardiology) - Available: Tue, Thu, Fri 10AM-4PM
    - Dr. Emily Rodriguez (Dermatology) - Available: Mon, Wed, Fri 1PM-6PM
    - Dr. David Kim (Orthopedics) - Available: Mon-Thu 8AM-3PM
    - Dr. Lisa Thompson (Pediatrics) - Available: Mon-Fri 9AM-4PM

    Response format:
    - If complete information provided: Confirm appointment
    - If partial information: Ask for missing details
    - If doctor unavailable: Suggest alternatives
    - Always be professional and helpful
    """
    
    user_prompt = f"""
    Appointment request: {request}
    
    Please help with this appointment request by:
    1. Analyzing what information is provided
    2. Identifying any missing information needed
    3. Providing appropriate scheduling assistance
    4. Confirming details if complete or requesting missing information
    
    Current date: {datetime.now().strftime('%Y-%m-%d')}
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm(messages)
        return response.content
    except Exception as e:
        return f"I apologize, but I'm having trouble processing your appointment request. Please try again or call our office directly. Error: {str(e)}"

# ----------------------------
# INTELLIGENT HEALTH ADVISOR
# ----------------------------
def intelligent_health_advisor(query: str) -> str:
    """
    General health advisor for queries that don't fit symptom analysis or scheduling
    """
    system_prompt = """You are a knowledgeable health advisor and wellness expert. Provide helpful, accurate health information 
    while always emphasizing the importance of consulting healthcare professionals for medical advice.
    
    You can help with:
    - General health questions and wellness advice
    - Preventive care information and health education
    - Lifestyle recommendations for better health
    - Medical terminology explanations
    - Nutrition and exercise guidance
    - Mental health and stress management tips
    - Understanding medical procedures and tests
    
    Guidelines:
    - Provide evidence-based information
    - Be encouraging and supportive
    - Always include appropriate medical disclaimers
    - Suggest when professional consultation is needed
    - Keep advice practical and actionable
    """
    
    user_prompt = f"""
    Health query: {query}
    
    Please provide helpful, informative response while maintaining appropriate medical disclaimers.
    Focus on being educational, supportive, and actionable.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm(messages)
        return response.content
    except Exception as e:
        return f"I'm here to help with health information. Please try rephrasing your question or consult with a healthcare professional. Error: {str(e)}"

# ----------------------------
# INTELLIGENT ROUTER
# ----------------------------
def intelligent_router(state: AppState) -> AppState:
    """
    Uses LLM to intelligently route user queries to appropriate agents
    """
    user_input = state["input"]
    
    system_prompt = """You are an intelligent router for a healthcare AI system. 
    Analyze the user's query and determine which agent should handle it.
    
    Available agents:
    1. symptom_agent - For symptom analysis, medical concerns, feeling unwell, pain, discomfort, illness, disease symptoms
    2. scheduler_agent - For appointment booking, scheduling, doctor visits, medical appointments, cancellations, rescheduling
    3. advisor_agent - For general health questions, wellness advice, health education, prevention, lifestyle, nutrition, exercise
    
    Examples:
    - "I have a headache and nausea" → symptom_agent
    - "Book appointment with Dr. Smith" → scheduler_agent  
    - "How to improve sleep quality?" → advisor_agent
    - "What causes high blood pressure?" → advisor_agent
    - "Chest pain and shortness of breath" → symptom_agent
    - "Cancel my appointment tomorrow" → scheduler_agent
    
    Respond with ONLY the agent name: symptom_agent, scheduler_agent, or advisor_agent
    """
    
    user_prompt = f"User query: {user_input}\n\nWhich agent should handle this?"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm(messages)
        route = response.content.strip().lower()
        
        # Ensure valid route
        if route not in ["symptom_agent", "scheduler_agent", "advisor_agent"]:
            route = "advisor_agent"  # Default to advisor for unclear queries
        
        return {**state, "route": route}
    except Exception as e:
        # Default to advisor agent if routing fails
        return {**state, "route": "advisor_agent"}

# ----------------------------
# Create Tools
# ----------------------------
symptom_tool = Tool.from_function(
    intelligent_symptom_analyzer, 
    name="SymptomAnalyzer", 
    description="Analyzes patient symptoms using advanced medical knowledge"
)

scheduler_tool = Tool.from_function(
    intelligent_appointment_scheduler, 
    name="AppointmentScheduler", 
    description="Handles appointment scheduling, booking, and management"
)

advisor_tool = Tool.from_function(
    intelligent_health_advisor, 
    name="HealthAdvisor", 
    description="Provides general health information and wellness guidance"
)

# ----------------------------
# Agent Node Functions
# ----------------------------
def symptom_agent(state: AppState) -> AppState:
    output = symptom_tool.invoke(state["input"])
    return {**state, "output": output}

def scheduler_agent(state: AppState) -> AppState:
    output = scheduler_tool.invoke(state["input"])
    return {**state, "output": output}

def advisor_agent(state: AppState) -> AppState:
    output = advisor_tool.invoke(state["input"])
    return {**state, "output": output}

# ----------------------------
# Build Graph
# ----------------------------
def create_healthcare_graph():
    builder = StateGraph(state_schema=AppState)
    
    # Add nodes
    builder.add_node("router", RunnableLambda(intelligent_router))
    builder.add_node("symptom_agent", RunnableLambda(symptom_agent))
    builder.add_node("scheduler_agent", RunnableLambda(scheduler_agent))
    builder.add_node("advisor_agent", RunnableLambda(advisor_agent))
    
    # Set entry point
    builder.set_entry_point("router")
    
    # Add conditional edges from router
    builder.add_conditional_edges("router", lambda state: state["route"], {
        "symptom_agent": "symptom_agent",
        "scheduler_agent": "scheduler_agent",
        "advisor_agent": "advisor_agent"
    })
    
    # Add edges to END
    builder.add_edge("symptom_agent", END)
    builder.add_edge("scheduler_agent", END)
    builder.add_edge("advisor_agent", END)
    
    return builder.compile()

# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.set_page_config(
        page_title="Intelligent Healthcare AI", 
        page_icon="🏥", 
        layout="wide"
    )
    
    # Header
    st.title("🏥 Intelligent Healthcare AI Assistant")
    st.markdown("*Powered by Advanced AI - Your Personal Health Companion*")
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main input area
        st.markdown("### 🗨️ Tell me about your health concern")
        user_input = st.text_area(
            "Describe your symptoms, book an appointment, or ask a health question:",
            placeholder="Examples:\n• I have a persistent headache and feel nauseous\n• Book me an appointment with Dr. Johnson for next Tuesday\n• How can I improve my sleep quality?\n• What are the signs of high blood pressure?",
            height=120,
            key="user_input"
        )
    
    with col2:
        # Sidebar information
        st.markdown("### 🎯 AI Capabilities")
        st.markdown("""
        **🩺 Symptom Analysis**
        - Comprehensive symptom evaluation
        - Severity assessment
        - Treatment recommendations
        
        **📅 Appointment Scheduling**
        - Natural language booking
        - Doctor availability checking
        - Appointment modifications
        
        **💡 Health Guidance**
        - Wellness advice
        - Preventive care tips
        - Health education
        """)
        
        st.markdown("### 👨‍⚕️ Available Doctors")
        st.markdown("""
        - **Dr. Sarah Johnson** - General Practice
        - **Dr. Michael Chen** - Cardiology  
        - **Dr. Emily Rodriguez** - Dermatology
        - **Dr. David Kim** - Orthopedics
        - **Dr. Lisa Thompson** - Pediatrics
        """)
    
    # Submit button
    if st.button("🚀 Get AI Response", type="primary", use_container_width=True):
        if user_input.strip():
            # Create graph
            graph = create_healthcare_graph()
            
            # Show processing
            with st.spinner("🤖 AI is analyzing your query..."):
                try:
                    # Invoke graph
                    result = graph.invoke({
                        "input": user_input, 
                        "output": "", 
                        "route": "",
                        "context": ""
                    })
                    
                    # Display result
                    st.success("✅ AI Response Generated!")
                    
                    # Show which agent handled the query
                    agent_names = {
                        "symptom_agent": "🩺 Symptom Analyzer",
                        "scheduler_agent": "📅 Appointment Scheduler", 
                        "advisor_agent": "💡 Health Advisor"
                    }
                    
                    st.info(f"**Handled by:** {agent_names.get(result['route'], 'Health Assistant')}")
                    
                    # Display response in a nice format
                    st.markdown("### 🤖 AI Response:")
                    st.markdown(result['output'])
                    
                    # Medical disclaimer
                    st.warning("""
                    ⚠️ **Important Medical Disclaimer:**  
                    This AI assistant provides general health information and should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical concerns.
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Please try again or contact support if the issue persists.")
        else:
            st.warning("⚠️ Please enter your health query or request.")
    
    # Example queries section
    st.markdown("---")
    st.markdown("### 💡 Example Queries You Can Try:")
    
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        st.markdown("**🩺 Symptom Analysis**")
        if st.button("Headache and nausea", key="ex1"):
            st.session_state.user_input = "I have a persistent headache and feel nauseous"
        if st.button("Chest pain", key="ex2"):
            st.session_state.user_input = "I'm experiencing chest pain and shortness of breath"
        if st.button("Fever and cough", key="ex3"):
            st.session_state.user_input = "My child has fever and persistent cough"
    
    with example_col2:
        st.markdown("**📅 Appointments**")
        if st.button("Book with Dr. Johnson", key="ex4"):
            st.session_state.user_input = "I need to book an appointment with Dr. Johnson for next Tuesday"
        if st.button("Cardiology consultation", key="ex5"):
            st.session_state.user_input = "I need a cardiology consultation this week"
        if st.button("Cancel appointment", key="ex6"):
            st.session_state.user_input = "I want to cancel my appointment tomorrow"
    
    with example_col3:
        st.markdown("**💡 Health Advice**")
        if st.button("Improve sleep", key="ex7"):
            st.session_state.user_input = "How can I improve my sleep quality?"
        if st.button("High blood pressure", key="ex8"):
            st.session_state.user_input = "What are the signs of high blood pressure?"
        if st.button("Healthy diet tips", key="ex9"):
            st.session_state.user_input = "Give me healthy diet recommendations"
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with ❤️ using LangChain, OpenAI GPT-4, and Streamlit*")

if __name__ == "__main__":
    main()