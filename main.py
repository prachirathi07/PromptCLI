import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from typing import Dict, List, Optional
import platform
import os
from dataclasses import dataclass

@dataclass
class CommandResponse:
    command: str
    explanation: str
    safety_warnings: List[str]
    os_specific_notes: str

class PromptCLI:
    def __init__(self, api_key: str):
        """Initialize PromptCLI with LangChain components."""
        os.environ["OPENAI_API_KEY"] = api_key
        self.llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5")
        self.safety_patterns = self.load_safety_patterns()
        self.command_chain = self._create_command_chain()
        self.explanation_chain = self._create_explanation_chain()

    def _create_command_chain(self) -> LLMChain:
        """Create LangChain chain for command generation."""
        system_template = """You are an expert CLI command generator for {os_type}.
        Your task is to convert natural language descriptions into accurate terminal commands.
        Follow these rules:
        1. Generate only the exact command without explanation
        2. Ensure commands are safe and follow best practices
        3. Use appropriate syntax for the specified OS
        4. Handle complex multi-step operations when needed
        5. Include proper error handling where applicable"""
        
        human_template = "Generate a command to: {task}"
        
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
        
        return LLMChain(llm=self.llm, prompt=chat_prompt)

    def _create_explanation_chain(self) -> LLMChain:
        """Create LangChain chain for command explanation."""
        system_template = """Analyze the given command and provide:
        1. A clear explanation of what the command does
        2. Any potential risks or considerations
        3. Expected output or behavior"""
        
        human_template = "Explain this command: {command}"
        
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
        
        return LLMChain(llm=self.llm, prompt=chat_prompt)

    @staticmethod
    def load_safety_patterns() -> Dict[str, List[str]]:
        """Load patterns for command safety checking."""
        return {
            "dangerous_commands": [
                "rm -rf", "del /F /S /Q", "format",
                "mkfs", ">", "dd if=",
            ],
            "sensitive_paths": [
                "/etc/passwd", "C:\\Windows\\System32",
                "/dev/sd", "C:\\Program Files"
            ]
        }

    def validate_command_safety(self, command: str) -> List[str]:
        """Check command against safety patterns."""
        warnings = []
        
        for pattern in self.safety_patterns["dangerous_commands"]:
            if pattern in command.lower():
                warnings.append(f"Warning: Command contains potentially dangerous operation: {pattern}")
        
        for path in self.safety_patterns["sensitive_paths"]:
            if path in command:
                warnings.append(f"Warning: Command attempts to access sensitive path: {path}")
        
        return warnings

    async def generate_command(self, task: str, os_type: str) -> CommandResponse:
        """Generate and validate a command using LangChain."""
        try:
            # Generate command
            with get_openai_callback() as cb:
                command_response = await self.command_chain.arun(
                    task=task,
                    os_type=os_type
                )
                command = command_response.strip()

                # Generate explanation
                explanation_response = await self.explanation_chain.arun(
                    command=command
                )

            # Validate command safety
            safety_warnings = self.validate_command_safety(command)
            
            return CommandResponse(
                command=command,
                explanation=explanation_response,
                safety_warnings=safety_warnings,
                os_specific_notes=f"Command generated for {os_type} environment. Token usage: {cb.total_tokens}"
            )
            
        except Exception as e:
            st.error(f"Error generating command: {str(e)}")
            return None

def main():
    st.set_page_config(
        page_title="PromptCLI - AI Command Generator",
        page_icon="📨",
        layout="wide"
    )
    
    st.title("📨 PromptCLI - AI Command Generator")
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # API key input
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    
    if not api_key:
        st.warning("Please enter your OpenAI API key to continue.")
        return
    
    # Initialize PromptCLI with LangChain
    cli = PromptCLI(api_key)
    
    # OS Selection
    os_options = {
        "Windows": "Windows",
        "Darwin": "MacOS",
        "Linux": "Linux"
    }
    selected_os = st.selectbox(
        "Select Operating System",
        options=list(os_options.values())
    )
    
    # Task input
    task = st.text_area("Describe what you want to do:", 
                        placeholder="E.g., 'list all files in current directory'")
    
    if st.button("Generate Command"):
        if task:
            with st.spinner("Generating command..."):
                # Use asyncio to run async function
                import asyncio
                result = asyncio.run(cli.generate_command(task, selected_os))
                
                if result:
                    # Display command with copy button
                    st.code(result.command, language="bash")
                    
                    # Display explanation
                    with st.expander("Command Explanation"):
                        st.write(result.explanation)
                    
                    # Display safety warnings if any
                    if result.safety_warnings:
                        st.warning("\n".join(result.safety_warnings))
                    
                    # Add to history
                    st.session_state.history.append({
                        "task": task,
                        "command": result.command,
                        "os": selected_os
                    })
    
    # Display command history
    if st.session_state.history:
        with st.expander("Command History"):
            for idx, item in enumerate(reversed(st.session_state.history)):
                st.text(f"Task: {item['task']}")
                st.code(item['command'], language="bash")
                st.text(f"OS: {item['os']}")
                st.divider()

if __name__ == "__main__":
    main()